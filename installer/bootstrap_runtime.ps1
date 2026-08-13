param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [switch]$InstallOllama,
    [switch]$OllamaOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$PythonExe = Join-Path $AppDir "runtime\python\python.exe"
$InstallerDir = Join-Path $AppDir "installer"
$LogDir = Join-Path $DataDir "logs"
$LogFile = Join-Path $LogDir "installer.log"
$OllamaModel = "qwen2.5:7b-instruct"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null
Set-Location -LiteralPath $AppDir

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $Arguments"
    }
}

function Find-Ollama {
    $command = Get-Command "ollama.exe" -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Enable-Ollama {
    Write-Host "Installing Jarvis Full / Ollama support..."
    Invoke-Native $PythonExe @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "-r", (Join-Path $InstallerDir "requirements-full.txt")
    )

    $ollama = Find-Ollama
    if (-not $ollama) {
        $setup = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $setup
        $signature = Get-AuthenticodeSignature -LiteralPath $setup
        if ($signature.Status -ne "Valid") {
            throw "Ollama installer signature is not valid: $($signature.Status)"
        }
        $process = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT", "/NORESTART" -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "Ollama installer failed with exit code $($process.ExitCode)"
        }
        $ollama = Find-Ollama
    }
    if (-not $ollama) { throw "Ollama executable was not found after installation" }

    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
    } catch {
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        $ready = $false
        foreach ($attempt in 1..30) {
            Start-Sleep -Milliseconds 500
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
                $ready = $true
                break
            } catch { }
        }
        if (-not $ready) { throw "Ollama server did not become ready" }
    }
    Invoke-Native $ollama @("pull", $OllamaModel)
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Bundled Python runtime is missing: $PythonExe"
    }
    $env:JARVIS_DATA_DIR = $DataDir
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

    if (-not $OllamaOnly) {
        Invoke-Native $PythonExe @("-m", "pip", "install", "--upgrade", "pip")

        $hasNvidia = $false
        try {
            $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
            $hasNvidia = [bool]($controllers | Where-Object Name -Match "NVIDIA")
        } catch { }
        $torchIndex = if ($hasNvidia) {
            "https://download.pytorch.org/whl/cu128"
        } else {
            "https://download.pytorch.org/whl/cpu"
        }
        Invoke-Native $PythonExe @(
            "-m", "pip", "install", "torch==2.11.0", "torchvision==0.26.0",
            "--index-url", $torchIndex
        )
        if ($hasNvidia) {
            $cudaReady = & $PythonExe -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
            if ($LASTEXITCODE -ne 0 -or $cudaReady.Trim() -ne "yes") {
                Write-Warning "CUDA PyTorch could not use this NVIDIA driver; falling back to CPU PyTorch."
                Invoke-Native $PythonExe @(
                    "-m", "pip", "install", "--force-reinstall", "torch==2.11.0",
                    "torchvision==0.26.0",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                )
            }
        }
        Invoke-Native $PythonExe @(
            "-m", "pip", "install", "--disable-pip-version-check",
            "-r", (Join-Path $InstallerDir "requirements-lite.txt")
        )
    }

    if ($InstallOllama -or $OllamaOnly) { Enable-Ollama }

    $parakeetModel = Join-Path $AppDir ".local\parakeet\models\nvidia--parakeet-tdt-0.6b-v3\model.safetensors"
    if (Test-Path -LiteralPath $parakeetModel) {
        $doctorJson = Join-Path $DataDir "doctor-report.json"
        & $PythonExe (Join-Path $AppDir "main.py") --doctor --json 1> $doctorJson
        if ($LASTEXITCODE -eq 2) {
            throw "Runtime Doctor found critical errors. See $doctorJson"
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Runtime Doctor failed with exit code $LASTEXITCODE"
        }
        Write-Host "JARVIS_INSTALL_READY doctor=$doctorJson"
    } else {
        Write-Warning "Parakeet model setup is required before first Jarvis start. Use the Start-menu shortcut."
        Write-Host "JARVIS_INSTALL_READY model_setup=required"
    }
} finally {
    Stop-Transcript | Out-Null
}
