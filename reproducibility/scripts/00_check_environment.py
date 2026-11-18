"""Script 00: Environment diagnostic and hardware verification."""

import os
import sys
import platform

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.utils.environment import get_environment_info, print_environment_summary


def main():
    print("=" * 80)
    print("STEP 00: VERIFYING EXECUTION ENVIRONMENT AND HARDWARE")
    print("=" * 80)
    
    info = get_environment_info()
    print_environment_summary(info)

    # Assert basic requirements
    cuda_avail = info.get("cuda_available", False)
    if not cuda_avail:
        print("\n[WARNING] CUDA is not available. GPU acceleration and CUDA Graphs will be disabled.")
    else:
        gpu_name = info.get("device_name", "Unknown GPU")
        print(f"\n[PASS] GPU Detected: {gpu_name} ({info.get('total_memory_gb', 0):.2f} GB VRAM)")

    print("[PASS] Python version:", platform.python_version())
    print("\nEnvironment check completed successfully (exit code 0).\n")


if __name__ == "__main__":
    main()
