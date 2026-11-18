"""Real GPU hardware trace replay engine on NVIDIA GeForce RTX 4060."""

from __future__ import annotations

import copy
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Ensure RTS_PROJECT root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase23b.models import build_model
from src.phase23b.bdd_dataset import get_bdd_transforms, load_bdd_split_samples, NUM_BDD_CLASSES
from .request import Request, RequestOutcome, RequestState
from .model_profiles import ModelProfile, load_model_profiles
from .priority_queue import FIFOQueue, EDFQueue, PriorityQueue
from .admission_control import AdmissionController
from .adaptive_controller import AdaptiveController
from .discrete_event_simulator import Policy, SimulationResult


class CUDAGraphRunner:
    """Wraps static CUDA Graph capture and replay for a 6-class BDD100K Vision Transformer."""

    def __init__(self, model: nn.Module, device: str = "cuda", num_classes: int = NUM_BDD_CLASSES):
        self.model = model.eval().to(device)
        self.device = device

        # Static input and output buffers
        self.static_input = torch.zeros(1, 3, 224, 224, device=device)
        self.static_output = torch.zeros(1, num_classes, device=device)

        # Warmup and capture
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
        """Copy input into static buffer, replay graph, and return output."""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()


class GPUTraceReplayer:
    """Replays request traces on actual RTX 4060 GPU with real BDD100K driving images."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quality_threshold: float = 45.0,
        firm_deadlines: bool = True,
        fixed_evit_model: str = "evit_small_k07_fuse",
    ):
        self.profiles = profiles
        self.device = device
        self.quality_threshold = quality_threshold
        self.firm_deadlines = firm_deadlines
        self.fixed_evit_model = fixed_evit_model

        self.admission_controller = AdmissionController(
            profiles=self.profiles,
            quality_threshold=self.quality_threshold,
        )
        self.adaptive_controller = AdaptiveController(
            profiles=self.profiles,
            quality_threshold=self.quality_threshold,
            firm_deadlines=self.firm_deadlines,
        )

        self.graph_runners: Dict[str, CUDAGraphRunner] = {}
        self.transform = get_bdd_transforms(is_train=False, img_size=224)
        self.test_samples: List[Tuple[str, int]] = []
        self.loaded_image_paths: set[str] = set()
        self.disk_image_count: int = 0
        self.synthetic_image_count: int = 0
        self.image_tensor_cache: Dict[str, torch.Tensor] = {}
        self._init_models_and_data()

    def _init_models_and_data(self):
        """Initialize models, capture CUDA graphs, and load BDD100K test sample references."""
        print("  [GPUTraceReplayer] Initializing models and capturing CUDA Graphs...", flush=True)
        for cfg_id, profile in self.profiles.items():
            model = build_model(cfg_id, num_classes=NUM_BDD_CLASSES, pretrained=False)
            ckpt_path = os.path.join(PROJECT_ROOT, profile.checkpoint_path)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = ckpt["state_dict"] if "state_dict" in ckpt else (ckpt["model"] if "model" in ckpt else ckpt)
                model.load_state_dict(state_dict)
            model.to(self.device).eval()

            if self.device == "cuda":
                self.graph_runners[cfg_id] = CUDAGraphRunner(model, device=self.device, num_classes=NUM_BDD_CLASSES)
            else:
                self.graph_runners[cfg_id] = None

        # Load canonical BDD100K test samples if available, fallback to smoke test list
        test_json = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations", "test_full.json")
        test_txt = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations", "test.txt")
        if os.path.isfile(test_json):
            import json
            try:
                with open(test_json, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                self.test_samples = [(it["image_name"], int(it["class_id"])) for it in manifest_data]
            except Exception:
                pass
        elif os.path.isfile(test_txt):
            self.test_samples = load_bdd_split_samples(test_txt)

    def resolve_image_path(self, image_name: Optional[str]) -> Optional[str]:
        """Resolve image file path on disk across BDD100K 100k test/val/train or flat directories."""
        if not image_name:
            return None
        fname = image_name if image_name.endswith(".jpg") else f"{image_name}.jpg"
        candidate_paths = [
            os.path.join(PROJECT_ROOT, "data", "bdd100k", "images", "100k", "test", fname),
            os.path.join(PROJECT_ROOT, "data", "bdd100k", "images", "100k", "val", fname),
            os.path.join(PROJECT_ROOT, "data", "bdd100k", "images", "100k", "train", fname),
            os.path.join(PROJECT_ROOT, "data", "bdd100k", "images", fname),
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                return p
        return None

    def load_image_tensor(self, image_idx: Optional[int], image_name: Optional[str]) -> torch.Tensor:
        """Load image tensor from disk or return synthetic tensor with tracking."""
        img_path = self.resolve_image_path(image_name)
        if img_path is not None:
            if img_path in self.image_tensor_cache:
                self.loaded_image_paths.add(img_path)
                self.disk_image_count += 1
                return self.image_tensor_cache[img_path]
            try:
                with Image.open(img_path) as img:
                    tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
                    self.image_tensor_cache[img_path] = tensor
                    self.loaded_image_paths.add(img_path)
                    self.disk_image_count += 1
                    return tensor
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}", flush=True)
        self.synthetic_image_count += 1
        return torch.randn(1, 3, 224, 224, device=self.device)

    def get_image_usage_stats(self) -> Dict[str, Any]:
        """Return audit statistics for images loaded during replay."""
        return {
            "unique_images_loaded": len(self.loaded_image_paths),
            "disk_image_loads": self.disk_image_count,
            "synthetic_image_loads": self.synthetic_image_count,
            "sample_image_paths": sorted(list(self.loaded_image_paths))[:5],
        }

    def replay_trace(
        self,
        policy: Policy,
        workload: List[Request],
        workload_name: str = "gpu_replay",
    ) -> SimulationResult:
        """Replay trace on actual hardware, recording real CUDA execution times and outputs."""
        trace = [copy.deepcopy(r) for r in workload]

        if policy in (Policy.FIFO_FIXED_DEIT, Policy.FIFO_FIXED_EVIT):
            queue: PriorityQueue = FIFOQueue()
        else:
            queue = EDFQueue()

        use_admission = policy in (Policy.EDF_ADMISSION_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC)
        is_adaptive = policy in (Policy.EDF_ADAPTIVE, Policy.PROPOSED_ADAPTIVE_AC)

        all_requests: List[Request] = []
        completed_requests: List[Request] = []
        model_counts = {cfg: 0 for cfg in self.profiles.keys()}

        virtual_clock_ms = 0.0
        gpu_busy_until_ms = 0.0
        total_gpu_busy_time_ms = 0.0

        # We step through arriving requests
        for req in trace:
            arrival_ms = req.arrival_time_ms

            # Advance clock and process any queued requests if GPU is idle
            while not queue.is_empty() and (virtual_clock_ms < arrival_ms):
                head = queue.pop()
                if virtual_clock_ms >= head.absolute_deadline_ms - 1e-6:
                    head.mark_expired(virtual_clock_ms, reason="Expired while queued")
                    all_requests.append(head)
                    continue

                # Select model
                if is_adaptive:
                    model_id, est_ms, slack = self.adaptive_controller.select_model(head, virtual_clock_ms)
                    if model_id is None:
                        head.mark_expired(virtual_clock_ms, reason="Infeasible at dispatch (firm deadline)")
                        all_requests.append(head)
                        continue
                else:
                    model_id = "deit_small_baseline" if "DeiT" in policy.value else self.fixed_evit_model
                    est_ms = self.profiles[model_id].get_latency_ms("p95", "cuda_graph")

                # Execute on GPU and measure real execution time
                img_tensor = self.load_image_tensor(head.image_idx, head.image_name)

                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)

                head.mark_dispatched(virtual_clock_ms, model_id, est_ms)
                model_counts[model_id] += 1

                start_ev.record()
                runner = self.graph_runners[model_id]
                logits = runner.run(img_tensor)
                end_ev.record()
                torch.cuda.synchronize()

                actual_exec_ms = float(start_ev.elapsed_time(end_ev))
                pred_label = int(torch.argmax(logits, dim=1).item())

                virtual_clock_ms += actual_exec_ms
                total_gpu_busy_time_ms += actual_exec_ms
                head.mark_completed(virtual_clock_ms, actual_exec_ms, pred_label)

                all_requests.append(head)
                completed_requests.append(head)

            # Now handle new arrival at arrival_ms
            virtual_clock_ms = max(virtual_clock_ms, arrival_ms)

            if use_admission:
                fixed_model = "deit_small_baseline" if policy == Policy.EDF_ADMISSION_FIXED_DEIT else None
                feasible = self.admission_controller.is_feasible(
                    request=req,
                    current_time_ms=virtual_clock_ms,
                    gpu_busy_until_ms=virtual_clock_ms,
                    queued_requests=queue.list_requests(),
                    fixed_model_id=fixed_model,
                )
                if not feasible:
                    req.mark_rejected(virtual_clock_ms, reason="Admission control rejected")
                    all_requests.append(req)
                    continue

            queue.push(req)

        # Flush remaining queued requests after last arrival
        while not queue.is_empty():
            head = queue.pop()
            if virtual_clock_ms >= head.absolute_deadline_ms - 1e-6:
                head.mark_expired(virtual_clock_ms, reason="Expired while queued")
                all_requests.append(head)
                continue

            if is_adaptive:
                model_id, est_ms, slack = self.adaptive_controller.select_model(head, virtual_clock_ms)
                if model_id is None:
                    head.mark_expired(virtual_clock_ms, reason="Infeasible at dispatch (firm deadline)")
                    all_requests.append(head)
                    continue
            else:
                model_id = "deit_small_baseline" if "DeiT" in policy.value else self.fixed_evit_model
                est_ms = self.profiles[model_id].get_latency_ms("p95", "cuda_graph")

            img_tensor = self.load_image_tensor(head.image_idx, head.image_name)

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            head.mark_dispatched(virtual_clock_ms, model_id, est_ms)
            model_counts[model_id] += 1

            start_ev.record()
            runner = self.graph_runners[model_id]
            logits = runner.run(img_tensor)
            end_ev.record()
            torch.cuda.synchronize()

            actual_exec_ms = float(start_ev.elapsed_time(end_ev))
            pred_label = int(torch.argmax(logits, dim=1).item())

            virtual_clock_ms += actual_exec_ms
            total_gpu_busy_time_ms += actual_exec_ms
            head.mark_completed(virtual_clock_ms, actual_exec_ms, pred_label)

            all_requests.append(head)
            completed_requests.append(head)

        # Aggregation
        num_offered = len(trace)
        num_rejected = sum(1 for r in all_requests if r.outcome == RequestOutcome.REJECTED)
        num_admitted = num_offered - num_rejected
        num_expired = sum(1 for r in all_requests if r.outcome == RequestOutcome.EXPIRED)
        num_completed = len(completed_requests)
        num_on_time = sum(1 for r in completed_requests if r.outcome == RequestOutcome.ON_TIME)
        num_late = sum(1 for r in completed_requests if r.outcome == RequestOutcome.LATE)

        on_time_ratio_offered = round(num_on_time / num_offered, 4) if num_offered > 0 else 0.0
        deadline_miss_ratio_completed = round(num_late / num_completed, 4) if num_completed > 0 else 0.0
        rejection_ratio = round(num_rejected / num_offered, 4) if num_offered > 0 else 0.0
        expiration_ratio = round(num_expired / num_offered, 4) if num_offered > 0 else 0.0

        resp_times = [r.response_time_ms for r in completed_requests if r.response_time_ms is not None]
        wait_times = [r.waiting_time_ms for r in completed_requests if r.waiting_time_ms is not None]

        mean_resp = round(float(np.mean(resp_times)), 3) if resp_times else 0.0
        median_resp = round(float(np.median(resp_times)), 3) if resp_times else 0.0
        p95_resp = round(float(np.percentile(resp_times, 95)), 3) if resp_times else 0.0
        p99_resp = round(float(np.percentile(resp_times, 99)), 3) if resp_times else 0.0

        mean_wait = round(float(np.mean(wait_times)), 3) if wait_times else 0.0
        median_wait = round(float(np.median(wait_times)), 3) if wait_times else 0.0

        makespan = max(virtual_clock_ms, 1e-6)
        gpu_util = round((total_gpu_busy_time_ms / makespan) * 100.0, 2)

        # Classification accuracy on actual test images
        correct_count = sum(1 for r in completed_requests if r.is_correct is True)
        actual_test_acc = round((correct_count / num_completed) * 100.0, 2) if num_completed > 0 else 0.0

        val_accs = [self.profiles[r.selected_model].val_top1_accuracy for r in completed_requests]
        mean_val_acc = round(float(np.mean(val_accs)), 2) if val_accs else 0.0

        return SimulationResult(
            policy=policy.value,
            workload_name=workload_name,
            num_offered=num_offered,
            num_admitted=num_admitted,
            num_rejected=num_rejected,
            num_expired=num_expired,
            num_completed=num_completed,
            num_completed_on_time=num_on_time,
            num_completed_late=num_late,
            on_time_ratio_offered=on_time_ratio_offered,
            deadline_miss_ratio_completed=deadline_miss_ratio_completed,
            rejection_ratio=rejection_ratio,
            expiration_ratio=expiration_ratio,
            mean_response_time_ms=mean_resp,
            median_response_time_ms=median_resp,
            p95_response_time_ms=p95_resp,
            p99_response_time_ms=p99_resp,
            mean_waiting_time_ms=mean_wait,
            median_waiting_time_ms=median_wait,
            gpu_busy_time_ms=round(total_gpu_busy_time_ms, 3),
            total_makespan_ms=round(makespan, 3),
            gpu_utilization_pct=gpu_util,
            mean_val_accuracy=mean_val_acc,
            mean_test_accuracy=actual_test_acc,
            model_selection_counts=model_counts,
            request_records=all_requests,
        )
