@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python Launcher is missing. Install 64-bit Python 3.10 with the py launcher and run this script again.
  exit /b 2
)
py -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>&1
if errorlevel 1 (
  echo ERROR: 64-bit Python 3.10 is missing. Install it and run this script again.
  exit /b 2
)
if not exist ".venv-jester\Scripts\python.exe" (
  py -3.10 -m venv .venv-jester || exit /b 1
)
".venv-jester\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m pip install -r training_workspace\requirements-jester.txt
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -c "import sys, torch; ok=torch.cuda.is_available(); print('Jester environment:', torch.__version__, '| CUDA:', ok, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); sys.exit(0 if ok else 2)"
if errorlevel 1 (
  echo ERROR: PyTorch cannot access the NVIDIA GPU. Update the NVIDIA driver and run this script again.
  exit /b 1
)
endlocal
