"""HTTP REST API and web application server for interactive RTS simulator.

Serves the front-end dashboard and provides REST endpoints for workload management,
discrete simulation (Mode A), and physical GPU execution (Mode B).
Zero external web-framework dependencies: runs purely on Python standard library.
"""

from __future__ import annotations

import cgi
import io
import json
import mimetypes
import os
import sys
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

from src.interactive_rts.workload_manager import WorkloadManager, BDD_CLASSES
from src.interactive_rts.gpu_inference import get_inference_engine
from src.interactive_rts.engine import InteractiveRTSEngine, SimulationConfig
from src.phase24.model_profiles import load_model_profiles

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


class InteractiveRTSHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving REST API endpoints and static assets."""

    workload_manager: WorkloadManager = WorkloadManager(upload_dir=UPLOADS_DIR)
    engine: InteractiveRTSEngine = InteractiveRTSEngine()

    def send_json(self, data: Any, status: int = 200):
        """Send formatted JSON response with CORS headers."""
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, message: str, status: int = 400):
        """Send JSON error response."""
        self.send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Route GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel_path)
            self._serve_file(file_path)
        elif path.startswith("/uploads/"):
            rel_path = path[len("/uploads/"):]
            file_path = os.path.join(UPLOADS_DIR, rel_path)
            self._serve_file(file_path)
        elif path == "/api/status":
            self._handle_get_status()
        elif path == "/api/models":
            self._handle_get_models()
        elif path == "/api/classes":
            self.send_json({"classes": list(BDD_CLASSES)})
        elif path == "/api/dataset/catalog":
            params = parse_qs(parsed.query)
            page = int(params.get("page", ["1"])[0])
            page_size = int(params.get("page_size", ["20"])[0])
            class_filter = params.get("class", [None])[0]
            catalog = self.workload_manager.get_catalog(page=page, page_size=page_size, class_filter=class_filter)
            self.send_json(catalog)
        elif path == "/api/workload":
            self.send_json({"requests": self.workload_manager.get_requests()})
        else:
            self.send_error(404, "Endpoint or file not found")

    def do_POST(self):
        """Route POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/upload":
                self._handle_upload()
            elif path == "/api/workload/add":
                data = self._read_json_body()
                req = self.workload_manager.add_request(
                    request_id=data.get("request_id"),
                    arrival_time_ms=float(data.get("arrival_time_ms", 0.0)),
                    relative_deadline_ms=float(data.get("relative_deadline_ms", 10.0)),
                    image_info=data.get("image_info"),
                    ground_truth_label=data.get("ground_truth_label"),
                )
                self.send_json({"success": True, "request": req, "staged_count": len(self.workload_manager.get_requests())})
            elif path == "/api/workload/remove":
                data = self._read_json_body()
                req_id = data.get("request_id")
                removed = self.workload_manager.remove_request(req_id)
                self.send_json({"success": removed, "staged_count": len(self.workload_manager.get_requests())})
            elif path == "/api/workload/clear":
                self.workload_manager.clear_requests()
                self.send_json({"success": True, "staged_count": 0})
            elif path == "/api/workload/preset":
                data = self._read_json_body()
                preset_name = data.get("preset_name", "nominal_loose")
                reqs = self.workload_manager.load_preset(preset_name)
                self.send_json({"success": True, "preset": preset_name, "requests": reqs})
            elif path == "/api/simulate":
                data = self._read_json_body()
                config = self._parse_simulation_config(data, default_mode="simulation")
                staged = self.workload_manager.get_requests()
                if not staged:
                    return self.send_error_json("Workload queue is empty. Add requests or load a preset first.")
                result = self.engine.run_workload(staged, config)
                self.send_json(result)
            elif path == "/api/gpu_execute":
                data = self._read_json_body()
                config = self._parse_simulation_config(data, default_mode="gpu")
                staged = self.workload_manager.get_requests()
                if not staged:
                    return self.send_error_json("Workload queue is empty. Add requests or load a preset first.")
                result = self.engine.run_workload(staged, config)
                self.send_json(result)
            elif path == "/api/compare":
                data = self._read_json_body()
                base_cfg = self._parse_simulation_config(data.get("baseline", {}), default_mode="simulation")
                prop_cfg = self._parse_simulation_config(data.get("proposed", {}), default_mode="simulation")
                staged = self.workload_manager.get_requests()
                if not staged:
                    return self.send_error_json("Workload queue is empty. Add requests or load a preset first.")
                result = self.engine.run_comparison(staged, base_cfg, prop_cfg)
                self.send_json(result)
            else:
                self.send_error(404, f"Unknown API endpoint: {path}")
        except Exception as e:
            traceback.print_exc()
            self.send_error_json(str(e), status=500)

    def _read_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON from request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw)

    def _parse_simulation_config(self, data: Dict[str, Any], default_mode: str = "simulation") -> SimulationConfig:
        """Parse dictionary into typed SimulationConfig."""
        return SimulationConfig(
            scheduling_policy=data.get("scheduling_policy", "EDF"),
            admission_control=bool(data.get("admission_control", True)),
            model_selection=data.get("model_selection", "adaptive"),
            execution_mode=data.get("execution_mode", default_mode),
            quality_threshold=float(data.get("quality_threshold", 45.0)),
            latency_metric=data.get("latency_metric", "p95"),
            timing_backend=data.get("timing_backend", "cuda_graph"),
            firm_deadlines=bool(data.get("firm_deadlines", True)),
        )

    def _handle_get_status(self):
        """Return system status including GPU hardware stats."""
        gpu_engine = get_inference_engine()
        device_info = gpu_engine.get_device_info()
        staged = self.workload_manager.get_requests()
        profiles = self.engine.profiles

        self.send_json({
            "status": "online",
            "device_info": device_info,
            "staged_request_count": len(staged),
            "available_models": list(profiles.keys()),
            "presets": [
                "periodic_high_tight", "bursty_traffic", "overload_stress", "nominal_loose",
                "scene1_arrival", "scene2_gpu_resource", "scene3_missed_deadlines",
                "scene4_fifo_vs_edf", "scene5_admission_control", "scene6_adaptive"
            ],
            "classes_count": len(BDD_CLASSES),
        })

    def _handle_get_models(self):
        """Return metadata for all 4 model profiles."""
        profiles = self.engine.profiles
        res = {}
        for cfg_id, p in profiles.items():
            res[cfg_id] = {
                "config_id": p.config_id,
                "display_name": p.display_name,
                "keep_rate": p.keep_rate,
                "parameters_m": p.parameters_m,
                "macs_g": p.macs_g,
                "val_top1_accuracy": p.val_top1_accuracy,
                "test_top1_accuracy": p.test_top1_accuracy,
                "cuda_graph_p95_ms": p.get_latency_ms("p95", "cuda_graph"),
                "cuda_graph_median_ms": p.get_latency_ms("median", "cuda_graph"),
                "throughput_fps": p.throughput_fps,
            }
        self.send_json({"profiles": res})

    def _handle_upload(self):
        """Handle multipart form file upload or base64 JSON payload."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
            )
            if "image" not in form:
                return self.send_error_json("No 'image' field in upload.")
            file_item = form["image"]
            filename = file_item.filename or "upload.jpg"
            file_bytes = file_item.file.read()
        else:
            # JSON upload with base64
            data = self._read_json_body()
            import base64
            b64_str = data.get("image_base64", "")
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            file_bytes = base64.b64decode(b64_str)
            filename = data.get("filename", "upload.jpg")

        upload_res = self.workload_manager.validate_and_save_upload(file_bytes, filename)
        upload_res["thumbnail_url"] = f"/uploads/{upload_res['thumbnail_filename']}"
        upload_res["image_url"] = f"/uploads/{upload_res['filename']}"
        self.send_json({"success": True, "image": upload_res})

    def _serve_file(self, file_path: str, forced_mime: Optional[str] = None):
        """Safely serve a static file from disk."""
        if not os.path.isfile(file_path):
            self.send_error(404, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        if forced_mime:
            mime_type = forced_mime
        elif not mime_type:
            mime_type = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    """Start the interactive RTS HTTP server."""
    server_address = (host, port)
    server = ThreadingHTTPServer(server_address, InteractiveRTSHandler)
    print(f"=================================================================")
    print(f"  Interactive RTS Vision Inference Simulator Server Running")
    print(f"  URL: http://{host}:{port}/")
    print(f"  Hardware: NVIDIA RTX 4060 GPU with CUDA Graph Acceleration")
    print(f"=================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port_arg)
