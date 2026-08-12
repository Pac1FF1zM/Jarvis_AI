@echo off
setlocal
set "ROOT=%~dp0"
set "BOOTSTRAP=%ROOT%experiments\parakeet\scripts\bootstrap.ps1"

if /I "%~1"=="--runtime" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -InstallRuntime
  exit /b %ERRORLEVEL%
)
if /I "%~1"=="--review-license" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -ReviewLicense
  exit /b %ERRORLEVEL%
)
if /I "%~1"=="--accept-license" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -AcceptLicense "%~2"
  exit /b %ERRORLEVEL%
)
if /I "%~1"=="--download" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -DownloadModel
  exit /b %ERRORLEVEL%
)
if /I "%~1"=="--status" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -Status
  exit /b %ERRORLEVEL%
)

echo Usage:
echo   SETUP_PARAKEET.cmd --runtime
echo   SETUP_PARAKEET.cmd --review-license
echo   SETUP_PARAKEET.cmd --accept-license CC-BY-4.0
echo   SETUP_PARAKEET.cmd --download
echo   SETUP_PARAKEET.cmd --status
exit /b 2
