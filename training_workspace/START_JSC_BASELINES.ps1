param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$CheckOnly,
    [switch]$Smoke,
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-training\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Training environment was not found. Run training_workspace\SETUP_RTX3090.ps1 first."
}

$Arguments = @(
    "-m", "training_workspace.run_jsc_baselines",
    "--device", $Device
)
if ($CheckOnly) { $Arguments += "--check-only" }
if ($Smoke) { $Arguments += "--smoke" }
if ($ResumeExisting) { $Arguments += "--resume-existing" }

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "JSC baseline process failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
