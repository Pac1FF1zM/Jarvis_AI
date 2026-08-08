@echo off
setlocal
set "JARVIS_DATA_DIR=%APPDATA%\Jarvis"
cd /d "%~dp0"
"%~dp0runtime\python\python.exe" "%~dp0jarvis_control.py" %*
if errorlevel 1 pause
