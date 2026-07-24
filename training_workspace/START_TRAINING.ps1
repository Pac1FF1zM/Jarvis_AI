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
    $Config = Join-Path $workspace "config.yaml"
}
$arguments = @("-m", "training_workspace.run", "--config", $Config)
if ($CheckOnly) {
    $arguments += "--check-only"
}
Push-Location $repo
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training runner failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
