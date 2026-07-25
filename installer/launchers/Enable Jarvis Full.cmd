@echo off
setlocal
set "JARVIS_DATA_DIR=%APPDATA%\Jarvis"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\bootstrap_runtime.ps1" -AppDir "%~dp0" -DataDir "%APPDATA%\Jarvis" -OllamaOnly
if errorlevel 1 (
  echo Не удалось включить Jarvis Full. Откройте %%APPDATA%%\Jarvis\logs\installer.log
  pause
  exit /b 1
)
echo Jarvis Full готов.
pause
