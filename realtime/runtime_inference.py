"""Runtime inference execution pipeline for deadline-aware Vision Transformers.

Handles:
  1. Safe GPU preloading of full_attention, moderate_dwconv, and aggressive_dwconv.
  2. Input preprocessing and host-to-device transfers.
  3. Synchronized GPU forward execution and nanosecond-resolution monotonic timing.
  4. Mode dispatch via the Phase 7 controller or fixed baseline strategies.
  5. Component-wise latency breakdown (controller, prep, transfer, model, post, e2e).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    MEAN,
    PATCH_SIZE,
    STD,
    RunConfig,
    SegmentationHead,
    build_backbone,
    features,
    require,
    set_seed,
    strict_load_backbone,
)
from benchmarks.benchmark_inference_modes import (
    MODES,
    load_finetuned_checkpoint,
    prepare_mode,
)
from realtime.controller import (
    ControllerError,
    ModeProfile,
    SelectionResult,
    load_benchmark_modes,
    select_mode,
)

DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json"
)
PHASE4_CHECKPOINT = PROJECT_ROOT / "outputs" / "pet_experiment" / "best_checkpoint.pt"
MODERATE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "finetune_moderate" / "multiepoch_10" / "best_checkpoint.pt"
)
AGGRESSIVE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "finetune_aggressive" / "multiepoch_10" / "best_checkpoint.pt"
)


@dataclass(frozen=True)
class RuntimeRequestResult:
    """Detailed trace record of a single runtime request execution."""

    request_id: str
    strategy: str
    deadline_ms: float
    workload_category: str
    selected_mode: str
    selected_miou: float
    configured_latency_bound_ms: float
    controller_latency_ms: float
    preprocessing_latency_ms: float
    transfer_latency_ms: float
    model_execution_ms: float
    model_cuda_event_ms: float
    postprocessing_latency_ms: float
    total_request_latency_ms: float
    predicted_empirical_feasibility: bool
    likely_deadline_miss: bool
    actual_deadline_hit: bool
    prediction_saved_path: str
    error_message: str = ""


class RuntimeModelRegistry:
    """Manages preloaded CUDA models for all three inference modes."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.models: dict[str, tuple[torch.nn.Module, SegmentationHead]] = {}
        self.details: dict[str, Any] = {}

    def preload_all(
        self,
        phase4_path: Path = PHASE4_CHECKPOINT,
        moderate_path: Path = MODERATE_CHECKPOINT,
        aggressive_path: Path = AGGRESSIVE_CHECKPOINT,
    ) -> None:
        """Construct, patch, strictly load, and move all three models to device."""
        require(self.device.type == "cuda", "CUDA GPU is required for RuntimeModelRegistry")
        require(phase4_path.is_file(), f"Phase 4 checkpoint missing: {phase4_path}")
        require(moderate_path.is_file(), f"Moderate checkpoint missing: {moderate_path}")
        require(aggressive_path.is_file(), f"Aggressive checkpoint missing: {aggressive_path}")

        p4_payload = torch.load(phase4_path, map_location="cpu", weights_only=False)
        cfg = RunConfig(**p4_payload["run_config"])
        split = p4_payload["split_manifest"]

        mod_payload = load_finetuned_checkpoint(
            moderate_path, "moderate", MODES["moderate_dwconv"], split
        )
        mod_payload["_path"] = str(moderate_path)

        agg_payload = load_finetuned_checkpoint(
            aggressive_path, "aggressive", MODES["aggressive_dwconv"], split
        )
        agg_payload["_path"] = str(aggressive_path)

        for mode_name, blocks in MODES.items():
            finetuned = (
                mod_payload
                if mode_name == "moderate_dwconv"
                else (agg_payload if mode_name == "aggressive_dwconv" else None)
            )
            backbone, head, details = prepare_mode(
                mode_name,
                blocks,
                cfg,
                p4_payload["head_state_dict"],
                self.device,
                finetuned,
            )
            backbone.eval()
            head.eval()
            self.models[mode_name] = (backbone, head)
            self.details[mode_name] = details

    def warmup(self, num_iterations: int = 5) -> None:
        """Run warm-up forward passes on all preloaded models to prime CUDA caches."""
        if num_iterations <= 0:
            return
        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.inference_mode():
            for mode_name, (backbone, head) in self.models.items():
                for _ in range(num_iterations):
                    feat = features(backbone, dummy)
                    _ = head(feat, (224, 224))
                torch.cuda.synchronize(self.device)


def preprocess_image(
    image_source: Image.Image | Path | str | torch.Tensor,
    target_size: int = 224,
) -> torch.Tensor:
    """Preprocess an input into a normalized float32 tensor of shape (1, 3, target_size, target_size)."""
    if isinstance(image_source, torch.Tensor):
        if image_source.dim() == 3:
            return image_source.unsqueeze(0)
        return image_source

    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    img_resized = TF.resize(img, [target_size, target_size], interpolation=Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    normalized = TF.normalize(tensor, MEAN, STD)
    return normalized.unsqueeze(0)


def execute_model_forward(
    backbone: torch.nn.Module,
    head: SegmentationHead,
    device_image: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Execute model forward pass with CUDA event and host monotonic timing.

    Returns:
      (prediction_mask_cpu, host_model_ms, cuda_event_ms)
    """
    device = device_image.device
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start_event.record()

    with torch.inference_mode():
        feat = features(backbone, device_image)
        logits = head(feat, (224, 224))
        preds = logits.argmax(dim=1)

    end_event.record()
    end_event.synchronize()
    t1 = time.perf_counter()

    host_ms = (t1 - t0) * 1000.0
    cuda_ms = start_event.elapsed_time(end_event)
    return preds.squeeze(0).cpu(), host_ms, cuda_ms


def process_runtime_request(
    request_id: str,
    strategy: str,
    deadline_ms: float,
    workload_category: str,
    image_source: Image.Image | Path | str | torch.Tensor,
    registry: RuntimeModelRegistry,
    modes_metadata: list[ModeProfile],
    latency_metric: str,
    save_prediction_path: Path | None = None,
) -> RuntimeRequestResult:
    """Execute a single runtime request end-to-end with strict timing isolation."""
    # 1. Start monotonic request timer
    t_start = time.perf_counter()

    # 2. Mode selection
    t_ctrl_0 = time.perf_counter()
    if strategy == "always_full_attention":
        selected_mode = "full_attention"
        full_prof = next(m for m in modes_metadata if m.name == "full_attention")
        selected_miou = full_prof.validation_miou
        bound_ms = full_prof.latency_for_metric(latency_metric)
        predicted_feasible = bound_ms <= deadline_ms
        likely_miss = not predicted_feasible
    elif strategy == "always_fastest":
        fastest_prof = min(
            modes_metadata,
            key=lambda m: (m.latency_for_metric(latency_metric), -m.validation_miou, m.name),
        )
        selected_mode = fastest_prof.name
        selected_miou = fastest_prof.validation_miou
        bound_ms = fastest_prof.latency_for_metric(latency_metric)
        predicted_feasible = bound_ms <= deadline_ms
        likely_miss = not predicted_feasible
    elif strategy == "adaptive_controller":
        res = select_mode(deadline_ms, modes_metadata, latency_metric)
        selected_mode = res.selected_mode
        selected_miou = res.selected_miou
        bound_ms = res.selected_empirical_latency_bound_ms
        predicted_feasible = not res.likely_deadline_miss
        likely_miss = res.likely_deadline_miss
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    t_ctrl_1 = time.perf_counter()
    ctrl_ms = (t_ctrl_1 - t_ctrl_0) * 1000.0

    # 3. Preprocessing
    t_prep_0 = time.perf_counter()
    image_tensor = preprocess_image(image_source)
    t_prep_1 = time.perf_counter()
    prep_ms = (t_prep_1 - t_prep_0) * 1000.0

    # 4. Host-to-device transfer
    t_trans_0 = time.perf_counter()
    image_cuda = image_tensor.to(registry.device, non_blocking=False)
    torch.cuda.synchronize(registry.device)
    t_trans_1 = time.perf_counter()
    trans_ms = (t_trans_1 - t_trans_0) * 1000.0

    # 5. Model forward pass on GPU
    backbone, head = registry.models[selected_mode]
    preds_cpu, model_host_ms, model_cuda_ms = execute_model_forward(
        backbone, head, image_cuda
    )

    # 6. Postprocessing / mask saving
    t_post_0 = time.perf_counter()
    saved_str = ""
    if save_prediction_path is not None:
        save_prediction_path.parent.mkdir(parents=True, exist_ok=True)
        mask_np = preds_cpu.byte().numpy()
        Image.fromarray(mask_np * 255, mode="L").save(save_prediction_path)
        saved_str = str(save_prediction_path.resolve())
    t_post_1 = time.perf_counter()
    post_ms = (t_post_1 - t_post_0) * 1000.0

    # 7. End-to-end completion timestamp
    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000.0

    # 8. Actual deadline satisfaction check (host monotonic duration <= deadline budget)
    actual_hit = e2e_ms <= deadline_ms

    return RuntimeRequestResult(
        request_id=request_id,
        strategy=strategy,
        deadline_ms=deadline_ms,
        workload_category=workload_category,
        selected_mode=selected_mode,
        selected_miou=selected_miou,
        configured_latency_bound_ms=bound_ms,
        controller_latency_ms=ctrl_ms,
        preprocessing_latency_ms=prep_ms,
        transfer_latency_ms=trans_ms,
        model_execution_ms=model_host_ms,
        model_cuda_event_ms=model_cuda_ms,
        postprocessing_latency_ms=post_ms,
        total_request_latency_ms=e2e_ms,
        predicted_empirical_feasibility=predicted_feasible,
        likely_deadline_miss=likely_miss,
        actual_deadline_hit=actual_hit,
        prediction_saved_path=saved_str,
    )
