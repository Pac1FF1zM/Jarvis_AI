@echo off
setlocal

set "JARVIS_ROOT=%~dp0"
set "JARVIS_PYTHON=%JARVIS_ROOT%venv\Scripts\python.exe"
set "SHADOW_SCRIPT=%JARVIS_ROOT%experiments\parakeet\scripts\shadow_test.py"

if not exist "%JARVIS_PYTHON%" (
  echo ERROR: Jarvis interpreter not found: "%JARVIS_PYTHON%"
  exit /b 2
)

"%JARVIS_PYTHON%" "%SHADOW_SCRIPT%" %*
exit /b %ERRORLEVEL%
