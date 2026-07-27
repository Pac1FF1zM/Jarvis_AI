param(
    [Parameter(Mandatory = $true)][string]$Videos,
    [Parameter(Mandatory = $true)][string]$Annotations,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $workspace
$python = Join-Path $repo ".venv-training\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Training venv not found. Run training_workspace\SETUP_RTX3090.ps1 first."
}
if (-not $Output) {
    $Output = Join-Path $workspace "gesture_data\ipn_manifest.jsonl"
}
Push-Location $repo
try {
    & $python -m training_workspace.build_gesture_manifest --videos $Videos --annotations $Annotations --output $Output
    if ($LASTEXITCODE -ne 0) { throw "IPN import failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
