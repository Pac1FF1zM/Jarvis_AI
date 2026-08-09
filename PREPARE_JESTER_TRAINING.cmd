@echo off
setlocal
cd /d "%~dp0"
call SETUP_JESTER_TRAINING.cmd
if errorlevel 1 exit /b 1
if not exist "data\raw\jester\metadata\jester_labels\labels.csv" (
  echo ERROR: Jester data is missing. Copy your licensed data\raw\jester and data\splits\jester while keeping it under your control, or run the licensed downloader first.
  exit /b 2
)
".venv-jester\Scripts\python.exe" -m src.jester.preflight --workers 0,4,8,12,16 --write-profile configs/jester_hardware.yaml
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m src.jester.rehearsal --config configs/jester_hardware.yaml
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m src.jester.quality_gate --config configs/jester_hardware.yaml
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m src.jester.doctor --config configs/jester_hardware.yaml --require-ready
if errorlevel 1 exit /b 1
echo Jester is ready for full tiny_3d_cnn training.
endlocal
