"""Workload and image request management for interactive RTS simulator."""

from __future__ import annotations

import os
import uuid
import shutil
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


DEFAULT_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))


class WorkloadManager:
    """Manages uploaded images, staging queues, and presets."""

    def __init__(self, upload_dir: Optional[str] = None, dataset_root: str = "data/bdd100k"):
        self.upload_dir = upload_dir or DEFAULT_UPLOAD_DIR
        self.dataset_root = dataset_root
        os.makedirs(self.upload_dir, exist_ok=True)
        self.staged_requests: List[Dict[str, Any]] = []
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

    def _resolve_image_path(self, img_name: str, split_hint: Optional[str] = None) -> str:
        """Find the image path across flat or 100k/10k subdirectories."""
        images_dir = os.path.join(self.dataset_root, "images")
        direct = os.path.join(images_dir, f"{img_name}.jpg")
        if os.path.isfile(direct):
            return direct
        if split_hint:
            cand = os.path.join(images_dir, "100k", split_hint, f"{img_name}.jpg")
            if os.path.isfile(cand):
                return cand
        for split in ("test", "val", "train"):
            cand = os.path.join(images_dir, "100k", split, f"{img_name}.jpg")
            if os.path.isfile(cand):
                return cand
        for split in ("test", "val", "train"):
            cand = os.path.join(images_dir, "10k", split, f"{img_name}.jpg")
            if os.path.isfile(cand):
                return cand
        return direct

    def validate_and_save_upload(self, file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
        """Validate uploaded file and save securely to upload directory."""
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f} MB.")

        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        safe_id = str(uuid.uuid4())[:8]
        safe_filename = f"upload_{safe_id}{ext}"
        target_path = os.path.join(self.upload_dir, safe_filename)

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        try:
            with Image.open(target_path) as img:
                img.verify()
            with Image.open(target_path) as img:
                width, height = img.size
                format_name = img.format
                # Create a small thumbnail for UI preview
                thumb_filename = f"thumb_{safe_id}.jpg"
                thumb_path = os.path.join(self.upload_dir, thumb_filename)
                thumb = img.convert("RGB")
                thumb.thumbnail((120, 120))
                thumb.save(thumb_path, "JPEG", quality=85)
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise ValueError(f"Corrupt or invalid image file: {str(e)}")

        return {
            "image_id": safe_id,
            "filename": safe_filename,
            "thumbnail_filename": thumb_filename,
            "file_path": target_path,
            "thumbnail_path": thumb_path,
            "width": width,
            "height": height,
            "format": format_name,
        }

    def add_request(
        self,
        request_id: Optional[str],
        arrival_time_ms: float,
        relative_deadline_ms: float,
        image_info: Optional[Dict[str, Any]] = None,
        ground_truth_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a request to the staged workload."""
        if arrival_time_ms < 0:
            raise ValueError("Arrival time must be non-negative.")
        if relative_deadline_ms <= 0:
            raise ValueError("Relative deadline must be strictly positive.")

        req_id = request_id.strip() if request_id and request_id.strip() else f"R_{len(self.staged_requests) + 1:03d}"

        # Prevent duplicate IDs
        existing_ids = {r["request_id"] for r in self.staged_requests}
        if req_id in existing_ids:
            req_id = f"{req_id}_{uuid.uuid4().hex[:4]}"

        gt_class_id = None
        if ground_truth_label:
            cleaned = ground_truth_label.strip()
            if cleaned in BDD_CLASSES:
                gt_class_id = BDD_CLASSES.index(cleaned)
            elif cleaned.isdigit() and 0 <= int(cleaned) < len(BDD_CLASSES):
                gt_class_id = int(cleaned)
                ground_truth_label = BDD_CLASSES[gt_class_id]

        req_dict = {
            "request_id": req_id,
            "arrival_time_ms": round(float(arrival_time_ms), 4),
            "relative_deadline_ms": round(float(relative_deadline_ms), 4),
            "absolute_deadline_ms": round(float(arrival_time_ms) + float(relative_deadline_ms), 4),
            "image_filename": image_info["filename"] if image_info else None,
            "thumbnail_filename": image_info["thumbnail_filename"] if image_info else None,
            "image_path": image_info["file_path"] if image_info else None,
            "ground_truth_label": ground_truth_label,
            "ground_truth_class_id": gt_class_id,
        }
        self.staged_requests.append(req_dict)
        # Sort staged workload by arrival time
        self.staged_requests.sort(key=lambda r: (r["arrival_time_ms"], r["request_id"]))
        return req_dict

    def remove_request(self, request_id: str) -> bool:
        """Remove a request from the staged workload."""
        before = len(self.staged_requests)
        self.staged_requests = [r for r in self.staged_requests if r["request_id"] != request_id]
        return len(self.staged_requests) < before

    def clear_requests(self):
        """Clear all staged requests."""
        self.staged_requests.clear()

    def get_requests(self) -> List[Dict[str, Any]]:
        """Return the current list of staged requests."""
        return list(self.staged_requests)

    def load_preset(self, preset_name: str) -> List[Dict[str, Any]]:
        """Load a curated preset workload using real BDD100K driving scene images."""
        self.clear_requests()
        annot_full = os.path.join(self.dataset_root, "annotations", "test_full.json")
        annot_path = annot_full if os.path.isfile(annot_full) else os.path.join(self.dataset_root, "annotations", "test.json")

        sample_pool = []
        if os.path.isfile(annot_path):
            try:
                import json
                with open(annot_path, "r", encoding="utf-8") as f:
                    samples_data = json.load(f)
                sample_pool = [(s["image_name"], s["scene"]) for s in samples_data]
            except Exception:
                sample_pool = []

        if not sample_pool:
            sample_pool = [
                ("c2461a5a-936bf379", "residential"),
                ("bcaf73c1-9433521c", "parking lot"),
                ("bf24495d-c9787981", "highway"),
                ("c57caddc-26ab6ed5", "tunnel"),
                ("c81ff2c0-420a8956", "city street"),
            ]

        # Verify which images exist on disk
        available_samples = []
        for img_name, scene in sample_pool:
            p = self._resolve_image_path(img_name, split_hint="test")
            if os.path.isfile(p):
                available_samples.append((img_name, scene, p))
                if len(available_samples) >= 50:
                    break

        if not available_samples:
            available_samples = [(f"sample_{i}", "city street", "") for i in range(10)]

        if preset_name == "periodic_high_tight":
            # 8 requests arriving every ~2.85 ms, tight deadlines [3.5, 5.0] ms
            delta_t = 2.85
            deadlines = [3.8, 4.2, 4.0, 3.6, 4.5, 4.0, 3.9, 4.3]
            for i in range(min(8, len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R_{i+1:02d}", i * delta_t, deadlines[i % len(deadlines)], img_name, scene, p)

        elif preset_name == "bursty_traffic":
            # 2 dense bursts (3 requests arriving at 1.0 ms intervals) followed by recovery gap
            arrivals = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 22.0]
            deadlines = [4.5, 5.0, 5.5, 4.5, 5.0, 5.5, 6.0, 5.0]
            for i in range(min(len(arrivals), len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R_burst_{i+1:02d}", arrivals[i], deadlines[i], img_name, scene, p)

        elif preset_name == "overload_stress":
            # 8 requests arriving every 1.8 ms (faster than EViT 0.5 latency 2.38 ms!)
            # Tests admission control rejection at arrival
            for i in range(min(8, len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R_over_{i+1:02d}", i * 1.8, 4.0, img_name, scene, p)

        elif preset_name == "nominal_loose":
            # 5 requests arriving comfortably with loose deadlines
            for i in range(min(5, len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R_nom_{i+1:02d}", i * 6.0, 12.0, img_name, scene, p)

        elif preset_name == "scene1_arrival":
            # 4 requests arriving sequentially to illustrate request inlet
            arrivals = [0.0, 2.5, 5.0, 7.5]
            deadlines = [6.0, 6.0, 6.0, 6.0]
            for i in range(min(len(arrivals), len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R{i+1:02d}", arrivals[i], deadlines[i], img_name, scene, p)

        elif preset_name == "scene2_gpu_resource":
            # 2 requests: R1 occupies GPU at t=0; R2 arrives at t=1.2ms and must wait in queue
            img0, scene0, p0 = available_samples[0]
            img1, scene1, p1 = available_samples[1 % len(available_samples)]
            self._add_preset_item("R01", 0.0, 8.0, img0, scene0, p0)
            self._add_preset_item("R02", 1.2, 7.0, img1, scene1, p1)

        elif preset_name == "scene3_missed_deadlines":
            # 5 requests arriving faster than DeiT execution time (arr every 1.5ms, exec ~3.78ms, deadline 4.5ms)
            # Produces queue congestion and missed deadlines
            arrivals = [0.0, 1.5, 3.0, 4.5, 6.0]
            for i in range(min(len(arrivals), len(available_samples))):
                img_name, scene, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R{i+1:02d}", arrivals[i], 4.5, img_name, scene, p)

        elif preset_name == "scene4_fifo_vs_edf":
            # Exact classic workload demonstrating EDF prioritization:
            # R0: arr=0.0, rel_d=6.0 (abs_d=6.0) -> runs [0.0, 3.78]
            # R1: arr=0.5, rel_d=10.0 (abs_d=10.5)
            # R2: arr=1.0, rel_d=6.5 (abs_d=7.5) -> earlier deadline than R1!
            img0, b0, p0 = available_samples[0]
            img1, b1, p1 = available_samples[1 % len(available_samples)]
            img2, b2, p2 = available_samples[2 % len(available_samples)]
            self._add_preset_item("R01", 0.0, 6.0, img0, b0, p0)
            self._add_preset_item("R02_loose", 0.5, 10.0, img1, b1, p1)
            self._add_preset_item("R03_urgent", 1.0, 6.5, img2, b2, p2)

        elif preset_name == "scene5_admission_control":
            # Overload scenario with 6 requests arriving every 1.6ms
            arrivals = [0.0, 1.6, 3.2, 4.8, 6.4, 8.0]
            for i in range(min(len(arrivals), len(available_samples))):
                img_name, breed, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R{i+1:02d}", arrivals[i], 4.2, img_name, breed, p)

        elif preset_name == "scene6_adaptive":
            # Requests with tight slacks that force controller to prune tokens:
            # R1: arr=0.0, d=3.0 (DeiT 3.78ms would miss; EViT-0.5 2.71ms completes on-time!)
            # R2: arr=3.2, d=3.6 (EViT-0.9 3.51ms fits!)
            # R3: arr=7.0, d=6.0 (DeiT fits!)
            arrivals = [0.0, 3.2, 7.0]
            deadlines = [3.0, 3.6, 6.0]
            for i in range(min(len(arrivals), len(available_samples))):
                img_name, breed, p = available_samples[i % len(available_samples)]
                self._add_preset_item(f"R{i+1:02d}", arrivals[i], deadlines[i], img_name, breed, p)

        else:
            raise ValueError(f"Unknown preset workload '{preset_name}'.")

        return self.get_requests()

    def get_catalog(
        self,
        page: int = 1,
        page_size: int = 20,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return paginated list of eligible BDD100K images from dataset."""
        if self._catalog_cache is None:
            annot_full = os.path.join(self.dataset_root, "annotations", "test_full.json")
            annot_path = annot_full if os.path.isfile(annot_full) else os.path.join(self.dataset_root, "annotations", "test.json")
            if not os.path.isfile(annot_path):
                annot_path = os.path.join(self.dataset_root, "annotations", "val.json")

            all_items = []
            if os.path.isfile(annot_path):
                import json
                with open(annot_path, "r", encoding="utf-8") as f:
                    all_items = json.load(f)
            self._catalog_cache = all_items

        all_items = self._catalog_cache
        if class_filter:
            all_items = [item for item in all_items if item.get("scene") == class_filter]

        total_count = len(all_items)
        start_idx = max(0, (page - 1) * page_size)
        end_idx = start_idx + page_size
        page_items = all_items[start_idx:end_idx]

        results = []
        for s in page_items:
            img_name = s["image_name"]
            img_path = self._resolve_image_path(img_name, split_hint="test")
            thumb_name = f"thumb_preset_{img_name}.jpg"
            thumb_path = os.path.join(self.upload_dir, thumb_name)
            if os.path.isfile(img_path) and not os.path.isfile(thumb_path):
                try:
                    with Image.open(img_path) as img:
                        t = img.convert("RGB")
                        t.thumbnail((120, 120))
                        t.save(thumb_path, "JPEG", quality=85)
                except Exception:
                    pass

            results.append({
                "image_name": img_name,
                "filename": f"{img_name}.jpg",
                "scene": s.get("scene"),
                "class_id": s.get("class_id"),
                "thumbnail_url": f"/uploads/{thumb_name}" if os.path.isfile(thumb_path) else None,
                "file_path": img_path if os.path.isfile(img_path) else None,
            })

        return {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": (total_count + page_size - 1) // page_size if total_count > 0 else 1,
            "class_filter": class_filter,
            "items": results,
            "classes": BDD_CLASSES,
        }

    def _add_preset_item(self, req_id: str, arr: float, d: float, img_name: str, breed: str, p: str):
        thumb_name = None
        if p and os.path.isfile(p):
            # Generate thumb if not exists
            safe_id = f"preset_{img_name}"
            thumb_name = f"thumb_{safe_id}.jpg"
            thumb_path = os.path.join(self.upload_dir, thumb_name)
            if not os.path.isfile(thumb_path):
                with Image.open(p) as img:
                    t = img.convert("RGB")
                    t.thumbnail((120, 120))
                    t.save(thumb_path, "JPEG", quality=85)

        self.add_request(
            request_id=req_id,
            arrival_time_ms=arr,
            relative_deadline_ms=d,
            image_info={
                "filename": f"{img_name}.jpg",
                "thumbnail_filename": thumb_name,
                "file_path": p,
            } if p else None,
            ground_truth_label=breed,
        )
