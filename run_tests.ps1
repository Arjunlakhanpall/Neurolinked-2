<#
.SYNOPSIS
  Setup a virtual environment, install CI test dependencies and run pytest.

.DESCRIPTION
  Creates and activates a local venv (.venv), installs minimal test requirements from
  requirements-ci.txt, installs a CPU-only PyTorch wheel, then runs pytest.

.PARAMETER Full
  If present, also installs full runtime requirements from requirements.txt

EXAMPLE
  .\run_tests.ps1
  .\run_tests.ps1 -Full
#>
param(
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
Write-Host "Starting test setup..."

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv"
    python -m venv .venv
} else {
    Write-Host ".venv already exists"
}

Write-Host "Activating .venv"
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing CI requirements"
pip install -r requirements-ci.txt

Write-Host "Installing CPU PyTorch wheel (this may take a moment)"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0+cpu

if ($Full) {
    Write-Host "Installing full runtime requirements (requirements.txt)"
    pip install -r requirements.txt
}

Write-Host "Running pytest"
pytest -q

Write-Host "Done."

