@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-jester\Scripts\python.exe" (
  py -3.10 -m venv .venv-jester || exit /b 1
)
".venv-jester\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m pip install -r training_workspace\requirements-jester.txt
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -c "import torch; print('Jester environment ready; CUDA:', torch.cuda.is_available(), torch.__version__)"
endlocal
