param(
    [switch]$CheckOnly,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $workspace
$python = Join-Path $repo ".venv-training\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Training venv not found. Run training_workspace\SETUP_RTX3090.ps1 first."
}
if (-not $Config) {
    $Config = Join-Path $workspace "gesture_config.yaml"
}
Push-Location $repo
try {
    $arguments = @("-m", "training_workspace.run_gesture_training", "--config", $Config)
    if ($CheckOnly) { $arguments += "--check-only" }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Gesture training runner failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
