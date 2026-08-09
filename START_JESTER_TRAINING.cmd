@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-jester\Scripts\python.exe" call SETUP_JESTER_TRAINING.cmd
if errorlevel 1 exit /b 1
set "JARVIS_JESTER_CONFIG=configs/jester_from_scratch.yaml"
if exist "configs\jester_hardware.yaml" set "JARVIS_JESTER_CONFIG=configs/jester_hardware.yaml"
".venv-jester\Scripts\python.exe" -m src.jester.doctor --config "%JARVIS_JESTER_CONFIG%" --require-ready
if errorlevel 1 exit /b 1
echo Starting full tiny_3d_cnn training. Existing latest.pt will resume automatically.
".venv-jester\Scripts\python.exe" -m src.jester.training train --config "%JARVIS_JESTER_CONFIG%" --model tiny_3d_cnn
set "JARVIS_JESTER_EXIT=%ERRORLEVEL%"
endlocal & exit /b %JARVIS_JESTER_EXIT%
