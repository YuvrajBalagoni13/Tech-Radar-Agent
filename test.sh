#!/usr/bin/env bash
# ==============================================================================
# Autonomous Tech Radar - Test Suite Runner
# Executes pytest using the uv virtual environment with clean environment isolation.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Ensure clean environment to prevent ROS system library hook conflicts
exec env -u PYTHONPATH "$VENV_DIR/bin/pytest" -v "$@"
