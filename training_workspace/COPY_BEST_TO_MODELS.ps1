param([switch]$Force)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $workspace
$source = Join-Path $workspace "export\jarvis_nlu_best.pt"
$sourceMetrics = Join-Path $workspace "export\jarvis_nlu_best.metrics.json"
$destination = Join-Path $repo "models\nlu_word_bigru_finetuned.pt"
$destinationMetrics = Join-Path $repo "models\nlu_word_bigru_finetuned.metrics.json"
if (-not (Test-Path $source)) {
    throw "Export not found. Complete START_TRAINING.ps1 successfully first."
}
if ((Test-Path $destination) -and -not $Force) {
    throw "$destination already exists. Use -Force only after reviewing report.json."
}
Copy-Item -LiteralPath $source -Destination $destination -Force:$Force
Copy-Item -LiteralPath $sourceMetrics -Destination $destinationMetrics -Force:$Force
Write-Host "Copied to $destination" -ForegroundColor Green
Write-Host "Set modules.nlu.model in config.yaml to models/nlu_word_bigru_finetuned.pt after local tests pass."
