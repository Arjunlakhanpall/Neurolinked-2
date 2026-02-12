#!/usr/bin/env bash
set -euo pipefail

FULL=0
if [[ "${1:-}" == "--full" || "${1:-}" == "-f" ]]; then
  FULL=1
fi

echo "Starting test setup..."

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment .venv"
  python3 -m venv .venv
else
  echo ".venv already exists"
fi

echo "Activating .venv"
# shellcheck source=/dev/null
source .venv/bin/activate

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing CI requirements"
pip install -r requirements-ci.txt

echo "Installing CPU PyTorch wheel (may take a moment)"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0+cpu

if [[ "$FULL" -eq 1 ]]; then
  echo "Installing full runtime requirements (requirements.txt)"
  pip install -r requirements.txt
fi

echo "Running pytest"
pytest -q

echo "Done."

