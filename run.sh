#!/usr/bin/env bash
# ==============================================================================
# Autonomous Tech Radar - Service Launcher
# Runs the FastAPI server using the uv virtual environment.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    echo "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python 3.12
    uv pip install -r "$SCRIPT_DIR/requirements.txt" --python "$VENV_DIR/bin/python"
fi

# Clean PYTHONPATH to avoid interference from system packages (e.g. ROS)
export PYTHONPATH="$SCRIPT_DIR"

echo "🚀 Starting Autonomous Tech Radar & JIT Learning Agent on http://localhost:8000"
exec "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --reload
