param([switch]$SkipToolBootstrap)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$InstallerDir = $PSScriptRoot
$CacheDir = Join-Path $InstallerDir "cache"
$OutputDir = Join-Path $InstallerDir "output"
$ToolsDir = Join-Path $InstallerDir ".tools"
$PythonUrl = "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe"
$PythonSha256 = "2A52993092A19CFDFFE126E2EEAC46A4265E25705614546604AD44988E040C0F"
$PythonInstaller = Join-Path $CacheDir "python-3.12.9-amd64.exe"
$InnoUrl = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe"
$InnoInstaller = Join-Path $CacheDir "innosetup-6.7.3.exe"

New-Item -ItemType Directory -Force -Path $CacheDir, $OutputDir, $ToolsDir | Out-Null

function Get-VerifiedDownload {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Sha256 = "",
        [string]$RequiredPublisher = ""
    )
    if (-not (Test-Path -LiteralPath $Destination)) {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Destination
    }
    if ($Sha256) {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash
        if ($actual -ne $Sha256) {
            Remove-Item -LiteralPath $Destination -Force
            throw "SHA-256 mismatch for $Destination (actual $actual)"
        }
    }
    if ($RequiredPublisher) {
        $signature = Get-AuthenticodeSignature -LiteralPath $Destination
        if ($signature.Status -ne "Valid" -or
            $signature.SignerCertificate.Subject -notmatch $RequiredPublisher) {
            throw "Invalid publisher signature for $Destination"
        }
    }
}

Get-VerifiedDownload -Url $PythonUrl -Destination $PythonInstaller -Sha256 $PythonSha256 -RequiredPublisher "Python Software Foundation"

$isccCandidates = @(
    (Join-Path $ToolsDir "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
)
$ISCC = $isccCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $ISCC -and -not $SkipToolBootstrap) {
    Get-VerifiedDownload -Url $InnoUrl -Destination $InnoInstaller -RequiredPublisher "Pyrsys B.V."
    $toolTarget = Join-Path $ToolsDir "Inno Setup 6"
    $toolLog = Join-Path $ToolsDir "inno-bootstrap.log"
    $arguments = '/VERYSILENT /CURRENTUSER /NORESTART /SUPPRESSMSGBOXES ' +
        '/PORTABLE=1 /DIR="' + $toolTarget + '" /LOG="' + $toolLog + '"'
    $process = Start-Process -FilePath $InnoInstaller -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Inno Setup bootstrap failed with exit code $($process.ExitCode); log: $toolLog"
    }
    $ISCC = Join-Path $toolTarget "ISCC.exe"
}
if (-not $ISCC -or -not (Test-Path -LiteralPath $ISCC)) {
    throw "ISCC.exe not found. Install Inno Setup 6.7 or run without -SkipToolBootstrap."
}

& $ISCC (Join-Path $InstallerDir "Jarvis.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }

$setup = Join-Path $OutputDir "Jarvis_Setup.exe"
if (-not (Test-Path -LiteralPath $setup)) { throw "Setup output is missing: $setup" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $setup).Hash
Write-Host "JARVIS_SETUP_READY path=$setup"
Write-Host "SHA256=$hash"
Write-Warning "Jarvis_Setup.exe is not code-signed yet; Windows SmartScreen may warn."
