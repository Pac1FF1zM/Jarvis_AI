param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11",
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $workspace
$venv = Join-Path $repo ".venv-training"

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $PythonLauncher $PythonVersion -m venv $venv
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaWheel"
& $python -m pip install -r (Join-Path $workspace "requirements-training.txt")
& $python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available(), 'CUDA GPU is not visible to PyTorch'"

Write-Host "Training environment is ready: $venv" -ForegroundColor Green
