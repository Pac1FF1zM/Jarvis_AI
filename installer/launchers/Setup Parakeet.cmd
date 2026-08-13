@echo off
setlocal
cd /d "%~dp0"

echo Installing the pinned Parakeet runtime...
powershell -NoProfile -ExecutionPolicy Bypass -File "experiments\parakeet\scripts\bootstrap.ps1" -InstallRuntime
if errorlevel 1 goto :failed

echo Downloading the CC-BY-4.0 license and pinned model card for review...
powershell -NoProfile -ExecutionPolicy Bypass -File "experiments\parakeet\scripts\bootstrap.ps1" -ReviewLicense
if errorlevel 1 goto :failed

start "Parakeet license" notepad ".local\parakeet\license-review\CC-BY-4.0.txt"
start "Parakeet model card" notepad ".local\parakeet\license-review\MODEL_CARD.md"
echo.
echo Review both opened files. To accept, type exactly: CC-BY-4.0
set /p "PARAKEET_ACCEPT=License identifier: "
if /i not "%PARAKEET_ACCEPT%"=="CC-BY-4.0" (
  echo License was not accepted. Model download cancelled.
  pause
  exit /b 2
)

powershell -NoProfile -ExecutionPolicy Bypass -File "experiments\parakeet\scripts\bootstrap.ps1" -AcceptLicense CC-BY-4.0
if errorlevel 1 goto :failed
powershell -NoProfile -ExecutionPolicy Bypass -File "experiments\parakeet\scripts\bootstrap.ps1" -DownloadModel
if errorlevel 1 goto :failed

echo Parakeet is ready. Run Jarvis Runtime Doctor to verify the installation.
pause
exit /b 0

:failed
echo Parakeet setup failed. Review the error above.
pause
exit /b 1
