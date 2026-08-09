@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-jester\Scripts\python.exe" call SETUP_JESTER_TRAINING.cmd
if errorlevel 1 exit /b 1
set "JARVIS_JESTER_CONFIG=configs/jester_from_scratch.yaml"
if exist "configs\jester_hardware.yaml" set "JARVIS_JESTER_CONFIG=configs/jester_hardware.yaml"
if not exist "reports\jester\benchmark.json" (
  echo ERROR: Candidate benchmark is missing. Run START_JESTER_BENCHMARK.cmd first.
  exit /b 2
)
".venv-jester\Scripts\python.exe" -m src.jester.doctor --config "%JARVIS_JESTER_CONFIG%" --require-ready
if errorlevel 1 exit /b 1
".venv-jester\Scripts\python.exe" -m src.jester.training train --config "%JARVIS_JESTER_CONFIG%"
endlocal
