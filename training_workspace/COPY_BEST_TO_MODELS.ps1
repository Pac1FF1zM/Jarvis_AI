param([switch]$Force)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $workspace
$source = Join-Path $workspace "export\jarvis_nlu_best.pt"
$sourceMetrics = Join-Path $workspace "export\jarvis_nlu_best.metrics.json"
$approvalPath = Join-Path $workspace "export\approved.json"
$destination = Join-Path $repo "models\nlu_manager_finetuned.pt"
$destinationMetrics = Join-Path $repo "models\nlu_manager_finetuned.metrics.json"
if (-not (Test-Path $source)) {
    throw "Export not found. Complete START_TRAINING.ps1 successfully first."
}
if (-not (Test-Path $approvalPath)) {
    throw "Export is not approved by the regression and holdout gates. Review the latest report.json."
}
$approval = Get-Content -LiteralPath $approvalPath -Raw | ConvertFrom-Json
if (-not $approval.approved) {
    throw "Export approval is false. Do not copy this checkpoint."
}
$actualHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne ([string]$approval.sha256).ToLowerInvariant()) {
    throw "Export hash differs from approved.json. Refusing to copy stale or modified weights."
}
if ((Test-Path $destination) -and -not $Force) {
    throw "$destination already exists. Use -Force only after reviewing report.json."
}
Copy-Item -LiteralPath $source -Destination $destination -Force:$Force
Copy-Item -LiteralPath $sourceMetrics -Destination $destinationMetrics -Force:$Force
Write-Host "Copied to $destination" -ForegroundColor Green
Write-Host "Set modules.nlu.model in config.yaml to models/nlu_manager_finetuned.pt after local tests pass."
