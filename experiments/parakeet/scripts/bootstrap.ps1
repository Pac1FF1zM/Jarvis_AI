[CmdletBinding()]
param(
    [switch]$InstallRuntime,
    [switch]$ReviewLicense,
    [string]$AcceptLicense = '',
    [switch]$DownloadModel,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$script = Join-Path $root 'experiments\parakeet\scripts\model_acquisition.py'
$requirements = Join-Path $root 'experiments\parakeet\manifests\requirements-parakeet.txt'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Jarvis interpreter not found: $python"
}
if ($InstallRuntime) {
    & $python -m pip install --disable-pip-version-check --requirement $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Parakeet runtime installation failed' }
    & $python -c "from transformers import AutoModelForTDT, AutoProcessor; print('Parakeet Transformers runtime ready')"
}
if ($ReviewLicense) { & $python $script ReviewLicense }
if ($AcceptLicense) { & $python $script AcceptLicense --accept-license $AcceptLicense }
if ($DownloadModel) { & $python $script DownloadModel }
if ($Status) { & $python $script Status }

if (-not $InstallRuntime -and -not $ReviewLicense -and -not $AcceptLicense -and -not $DownloadModel -and -not $Status) {
    Write-Host 'Use -InstallRuntime, -ReviewLicense, -AcceptLicense CC-BY-4.0, -DownloadModel, or -Status.'
}
