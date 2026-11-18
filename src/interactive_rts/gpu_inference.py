"""Physical GPU inference module for interactive RTS simulator.

Provides cached model loading, image preprocessing, high-precision CUDA event
timing, and label prediction using fine-tuned DeiT and EViT checkpoints.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from src.phase23b.models import build_model
from src.interactive_rts.workload_manager import BDD_CLASSES
from src.phase24.model_profiles import load_model_profiles, ModelProfile

CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "phase23b", "checkpoints")
)

# Standard evaluation transform
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CUDAGraphRunner:
    """Static CUDA Graph capture and replay for sub-4ms GPU inference on RTX 4060."""

    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = len(BDD_CLASSES)):
        self.model = model.eval().to(device)
        self.device = device
        self.static_input = torch.zeros(1, 3, 224, 224, device=device)
        self.static_output = torch.zeros(1, num_classes, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(15):
                self.static_output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Copy input into static buffer, replay CUDA graph, and return output."""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()


class GPUInferenceEngine:
    """Manages GPU execution and inference for the four RTS models."""

    def __init__(self, device: Optional[str] = None, use_cuda_graph: bool = True):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_cuda_graph = use_cuda_graph and (self.device.type == "cuda")
        self.models: Dict[str, nn.Module] = {}
        self.graph_runners: Dict[str, CUDAGraphRunner] = {}
        self.profiles: Dict[str, ModelProfile] = load_model_profiles()
        self.classes: List[str] = list(BDD_CLASSES)

    def get_device_info(self) -> Dict[str, Any]:
        """Return device capabilities and memory statistics."""
        is_cuda = self.device.type == "cuda"
        info = {
            "device": str(self.device),
            "is_cuda": is_cuda,
            "device_name": torch.cuda.get_device_name(self.device) if is_cuda else "CPU",
            "vram_total_mb": 0.0,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
        }
        if is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            info["vram_total_mb"] = round(props.total_memory / (1024 * 1024), 2)
            info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
            info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2)
        return info

    def load_model(self, config_id: str) -> nn.Module:
        """Load a specific model checkpoint into cache on GPU/CPU."""
        if config_id in self.models:
            return self.models[config_id]

        profile = self.profiles.get(config_id)
        if profile and profile.checkpoint_path and os.path.isfile(profile.checkpoint_path):
            ckpt_path = profile.checkpoint_path
        else:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{config_id}_bdd_best.pth")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {config_id} at {ckpt_path}")

        model = build_model(config_id, num_classes=len(self.classes))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Warmup forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            _ = model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.models[config_id] = model
        if self.use_cuda_graph:
            self.graph_runners[config_id] = CUDAGraphRunner(model, self.device, num_classes=len(self.classes))
        return model

    def preload_all(self):
        """Preload all 4 model configurations."""
        for cfg_id in ["deit_small_baseline", "evit_small_k09_fuse", "evit_small_k07_fuse", "evit_small_k05_fuse"]:
            self.load_model(cfg_id)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image file into a normalized (1, 3, 224, 224) tensor."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        img = Image.open(image_path).convert("RGB")
        tensor = EVAL_TRANSFORM(img)
        return tensor.unsqueeze(0).to(self.device)

    def run_inference(
        self,
        config_id: str,
        image_path: Optional[str] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run physical inference and measure accurate execution latency.

        Returns:
            Dict containing:
                - latency_ms: measured GPU/CPU latency in milliseconds
                - predicted_class_idx: int top-1 index
                - predicted_class_name: str BDD100K driving scene class name
                - confidence: float probability (0-1)
                - top3: list of (class_name, confidence)
        """
        model = self.load_model(config_id)

        if image_tensor is not None:
            tensor = image_tensor.to(self.device)
        elif image_path is not None:
            tensor = self.preprocess_image(image_path)
        else:
            # Synthetic tensor if no image provided
            tensor = torch.zeros(1, 3, 224, 224, device=self.device)

        if self.use_cuda_graph and config_id in self.graph_runners:
            runner = self.graph_runners[config_id]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            logits = runner.run(tensor)
            end_event.record()
            torch.cuda.synchronize(self.device)
            latency_ms = start_event.elapsed_time(end_event)
        elif self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                logits = model(tensor)
            end_event.record()
            torch.cuda.synchronize(self.device)
            latency_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(3, len(self.classes)))

        top1_idx = int(top_indices[0].item())
        top1_name = self.classes[top1_idx] if top1_idx < len(self.classes) else f"class_{top1_idx}"
        confidence = float(top_probs[0].item())

        top3 = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            name = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            top3.append({"class_name": name, "confidence": round(p, 4), "class_idx": idx})

        return {
            "config_id": config_id,
            "latency_ms": round(latency_ms, 3),
            "predicted_class_idx": top1_idx,
            "predicted_class_name": top1_name,
            "confidence": round(confidence, 4),
            "top3": top3,
            "device": str(self.device),
        }


# Singleton instance
_INFERENCE_ENGINE: Optional[GPUInferenceEngine] = None


def get_inference_engine() -> GPUInferenceEngine:
    """Retrieve or initialize the singleton GPU inference engine."""
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        _INFERENCE_ENGINE = GPUInferenceEngine()
    return _INFERENCE_ENGINE
