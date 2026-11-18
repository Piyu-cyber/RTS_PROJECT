"""Phase 24B Physical GPU Trace Replayer and Empirical Accuracy Evaluator.

Executes captured CUDA Graphs on the NVIDIA RTX 4060 Laptop GPU using real
BDD100K driving scene test images, measuring real GPU execution latency via CUDA events
and recording prediction-level correctness.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.phase24.model_profiles import ModelProfile
from src.phase24.request import Request, RequestOutcome
from src.phase24.priority_queue import FIFOQueue, EDFQueue
from src.phase24.admission_control import AdmissionController
from src.phase24.adaptive_controller import AdaptiveController
from src.phase24.discrete_event_simulator import Policy
from src.phase23b.models import build_model
from src.phase23b.bdd_dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, NUM_BDD_CLASSES
from src.phase24b.experiment_engine import CanonicalResult


class PhysicalCUDAGraphRunner:
    """Wrapper managing static GPU memory and replay for one DeiT/EViT configuration."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.static_input = torch.zeros(1, 3, 224, 224, device=self.device)

        # Warmup iterations to initialize internal state
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(10):
                _ = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()


class PhysicalGPUReplayer:
    """Replays real-time request traces on physical NVIDIA RTX 4060 GPU."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        dataset_root: str = "data/bdd100k",
        device: str = "cuda",
        quality_threshold: float = 45.0,
    ):
        self.profiles = profiles
        self.dataset_root = dataset_root
        self.device = device if torch.cuda.is_available() else "cpu"
        self.quality_threshold = quality_threshold

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

        self.admission_controller = AdmissionController(
            profiles=profiles,
            latency_metric="p95",
            backend="cuda_graph",
            quality_threshold=quality_threshold,
        )
        self.adaptive_controller = AdaptiveController(
            profiles=profiles,
            latency_metric="p95",
            backend="cuda_graph",
            quality_threshold=quality_threshold,
            firm_deadlines=True,
        )

        self.image_cache: Dict[str, torch.Tensor] = {}
        self.graph_runners: Dict[str, PhysicalCUDAGraphRunner] = {}
        self._init_cuda_graphs()

    def _init_cuda_graphs(self):
        """Build and capture CUDA Graphs for all four candidate models."""
        configs = [
            "deit_small_baseline",
            "evit_small_k09_fuse",
            "evit_small_k07_fuse",
            "evit_small_k05_fuse",
        ]

        print("  [PhysicalGPUReplayer] Loading checkpoints and capturing CUDA Graphs...", flush=True)
        for cfg_id in configs:
            model = build_model(cfg_id, num_classes=NUM_BDD_CLASSES, pretrained=False)
            ckpt_path = self.profiles[cfg_id].checkpoint_path
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else (ckpt["model"] if "model" in ckpt else ckpt)
            model.load_state_dict(state_dict, strict=True)
            self.graph_runners[cfg_id] = PhysicalCUDAGraphRunner(model, device=self.device)
        print("  [PhysicalGPUReplayer] All CUDA Graphs ready.", flush=True)

    def load_image_tensor(self, image_name: Optional[str]) -> torch.Tensor:
        """Load and cache preprocessed image tensor."""
        if not image_name:
            return torch.zeros(1, 3, 224, 224, device=self.device)

        if image_name in self.image_cache:
            return self.image_cache[image_name]

        img_path = os.path.join(self.dataset_root, "images", f"{image_name}.jpg")
        if not os.path.isfile(img_path):
            tensor = torch.zeros(1, 3, 224, 224, device=self.device)
        else:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        self.image_cache[image_name] = tensor
        return tensor

    def replay_trace(
        self,
        policy: Policy,
        workload: List[Request],
        workload_name: str,
        workload_category: str = "replay",
        load_level: str = "replay",
        deadline_tightness: str = "replay",
        seed: int = 42,
    ) -> CanonicalResult:
        """Execute physical GPU replay of request trace."""
        is_edf = policy in (
            Policy.EDF_FIXED_DEIT,
            Policy.EDF_FIXED_EVIT,
            Policy.EDF_ADMISSION_FIXED_DEIT,
            Policy.EDF_ADAPTIVE,
            Policy.PROPOSED_ADAPTIVE_AC,
        )
        queue = EDFQueue() if is_edf else FIFOQueue()
        use_admission = policy in (Policy.EDF_ADMISSION_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC)
        is_adaptive = policy in (Policy.EDF_ADAPTIVE, Policy.PROPOSED_ADAPTIVE_AC)

        all_requests: List[Request] = []
        completed_requests: List[Request] = []
        model_counts = {"deit_small_baseline": 0, "evit_small_k09_fuse": 0, "evit_small_k07_fuse": 0, "evit_small_k05_fuse": 0}

        virtual_clock_ms = 0.0
        total_gpu_busy_time_ms = 0.0
        last_event_time = 0.0
        time_weighted_q_area = 0.0
        max_q_len = 0

        def update_queue_tracking(now: float):
            nonlocal last_event_time, time_weighted_q_area, max_q_len
            dt = max(0.0, now - last_event_time)
            q_size = len(queue)
            time_weighted_q_area += q_size * dt
            if q_size > max_q_len:
                max_q_len = q_size
            last_event_time = now

        for req in workload:
            arrival_ms = req.arrival_time_ms

            # Step 1: Drain queue if GPU is idle before next arrival
            while not queue.is_empty() and (virtual_clock_ms < arrival_ms):
                head = queue.pop()
                update_queue_tracking(virtual_clock_ms)

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
                    model_id = "deit_small_baseline" if "DeiT" in policy.value else "evit_small_k07_fuse"
                    est_ms = self.profiles[model_id].get_latency_ms("p95", "cuda_graph")

                # Real GPU execution
                img_tensor = self.load_image_tensor(head.image_name)
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)

                head.mark_dispatched(virtual_clock_ms, model_id, est_ms)
                model_counts[model_id] += 1

                start_ev.record()
                logits = self.graph_runners[model_id].run(img_tensor)
                end_ev.record()
                torch.cuda.synchronize()

                actual_exec_ms = float(start_ev.elapsed_time(end_ev))
                pred_label = int(torch.argmax(logits, dim=1).item())

                virtual_clock_ms += actual_exec_ms
                total_gpu_busy_time_ms += actual_exec_ms
                head.mark_completed(virtual_clock_ms, actual_exec_ms, pred_label)

                all_requests.append(head)
                completed_requests.append(head)

            # Step 2: Handle new arrival at arrival_ms
            virtual_clock_ms = max(virtual_clock_ms, arrival_ms)
            update_queue_tracking(virtual_clock_ms)

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
            update_queue_tracking(virtual_clock_ms)

        # Step 3: Drain remaining queue after all arrivals have completed
        while not queue.is_empty():
            head = queue.pop()
            update_queue_tracking(virtual_clock_ms)

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
                model_id = "deit_small_baseline" if "DeiT" in policy.value else "evit_small_k07_fuse"
                est_ms = self.profiles[model_id].get_latency_ms("p95", "cuda_graph")

            img_tensor = self.load_image_tensor(head.image_name)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            head.mark_dispatched(virtual_clock_ms, model_id, est_ms)
            model_counts[model_id] += 1

            start_ev.record()
            logits = self.graph_runners[model_id].run(img_tensor)
            end_ev.record()
            torch.cuda.synchronize()

            actual_exec_ms = float(start_ev.elapsed_time(end_ev))
            pred_label = int(torch.argmax(logits, dim=1).item())

            virtual_clock_ms += actual_exec_ms
            total_gpu_busy_time_ms += actual_exec_ms
            head.mark_completed(virtual_clock_ms, actual_exec_ms, pred_label)

            all_requests.append(head)
            completed_requests.append(head)

        # Compute aggregate metrics
        num_offered = len(workload)
        num_rejected = sum(1 for r in all_requests if r.outcome == RequestOutcome.REJECTED)
        num_admitted = num_offered - num_rejected
        num_expired = sum(1 for r in all_requests if r.outcome == RequestOutcome.EXPIRED)
        num_completed = len(completed_requests)
        num_on_time = sum(1 for r in completed_requests if r.outcome == RequestOutcome.ON_TIME)
        num_late = sum(1 for r in completed_requests if r.outcome == RequestOutcome.LATE)

        on_time_ratio_offered = round(num_on_time / num_offered, 4) if num_offered > 0 else 0.0
        on_time_ratio_accepted = round(num_on_time / num_admitted, 4) if num_admitted > 0 else 0.0
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
        mean_q_len = round(time_weighted_q_area / makespan, 3)

        # Empirical accuracy over completed requests
        correct_count = sum(1 for r in completed_requests if r.is_correct is True)
        actual_test_acc = round((correct_count / num_completed) * 100.0, 2) if num_completed > 0 else 0.0

        # Synthetic expected accuracy for comparison
        if completed_requests:
            val_accs = [self.profiles[r.selected_model].val_top1_accuracy for r in completed_requests]
            test_accs = [self.profiles[r.selected_model].test_top1_accuracy for r in completed_requests]
            mean_val_acc = round(float(np.mean(val_accs)), 2)
            synthetic_test_acc = round(float(np.mean(test_accs)), 2)
        else:
            mean_val_acc = 0.0
            synthetic_test_acc = 0.0

        res = CanonicalResult(
            policy=policy.value,
            workload_name=workload_name,
            workload_category=workload_category,
            load_level=load_level,
            deadline_tightness=deadline_tightness,
            seed=seed,
            experiment_type="GPU_REPLAY",
            num_offered=num_offered,
            num_admitted=num_admitted,
            num_rejected=num_rejected,
            num_expired=num_expired,
            num_completed=num_completed,
            num_completed_on_time=num_on_time,
            num_completed_late=num_late,
            on_time_ratio_offered=on_time_ratio_offered,
            on_time_ratio_accepted=on_time_ratio_accepted,
            deadline_miss_ratio_completed=deadline_miss_ratio_completed,
            rejection_ratio=rejection_ratio,
            expiration_ratio=expiration_ratio,
            mean_response_time_ms=mean_resp,
            median_response_time_ms=median_resp,
            p95_response_time_ms=p95_resp,
            p99_response_time_ms=p99_resp,
            mean_waiting_time_ms=mean_wait,
            median_waiting_time_ms=median_wait,
            mean_queue_length=mean_q_len,
            max_queue_length=max_q_len,
            total_gpu_busy_time_ms=round(total_gpu_busy_time_ms, 3),
            total_makespan_ms=round(makespan, 3),
            gpu_utilization_pct=gpu_util,
            mean_val_accuracy=mean_val_acc,
            synthetic_expected_test_accuracy=synthetic_test_acc,
            empirical_test_accuracy=actual_test_acc,
            count_deit_small=model_counts["deit_small_baseline"],
            count_evit_k09=model_counts["evit_small_k09_fuse"],
            count_evit_k07=model_counts["evit_small_k07_fuse"],
            count_evit_k05=model_counts["evit_small_k05_fuse"],
            request_records=all_requests,
        )

        res.verify_conservation()
        return res
