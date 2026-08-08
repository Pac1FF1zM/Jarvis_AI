@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\pythonw.exe" (
  start "" "venv\Scripts\pythonw.exe" "jarvis_control.py"
) else (
  python "jarvis_control.py"
)
