# Install dependencies for Confidence-Weighted Curriculum RL
# Run from project root: .\scripts\install.ps1

$ErrorActionPreference = "Stop"
$venv = Join-Path $PSScriptRoot ".." ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating .venv..."
    python -m venv $venv
}

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip"
    exit $LASTEXITCODE
}

Write-Host "Installing PyTorch with CUDA 13.0..."
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install PyTorch"
    exit $LASTEXITCODE
}

Write-Host "Installing requirements..."
& $pip install -r (Join-Path $PSScriptRoot "..\requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install requirements"
    exit $LASTEXITCODE
}

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
