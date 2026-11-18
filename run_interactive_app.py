"""Launcher for Interactive Real-Time Vision Inference Simulator.

Run with:
    .\.venv_evit\Scripts\python run_interactive_app.py [port]

Default port: 8000
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.interactive_rts.app import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port)
