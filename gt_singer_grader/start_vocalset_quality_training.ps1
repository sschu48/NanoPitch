param(
    [string]$VocalSetRoot = ".\gt_singer_grader\data\VocalSet",
    [string]$OutputDir = ".\gt_singer_grader\runs\vocalset_quality",
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$MaxRecords = 0,
    [switch]$Download,
    [switch]$Extract
)

$repoRoot = Split-Path -Parent $PSScriptRoot
function Resolve-PythonCommand {
    $candidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $repoRoot) ".venvs\nanopitch\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $repoRoot) ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return "python"
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return "py"
    }

    throw "No Python runtime was found."
}

$python = Resolve-PythonCommand

Push-Location $repoRoot
try {
    if ($Download) {
        New-Item -ItemType Directory -Force -Path $VocalSetRoot | Out-Null
        $zipPath = Join-Path $VocalSetRoot "VocalSet1-2.zip"
        & curl.exe -L -C - -o $zipPath "https://zenodo.org/records/1442513/files/VocalSet1-2.zip?download=1"
        if ($LASTEXITCODE -ne 0) {
            throw "VocalSet download failed with exit code $LASTEXITCODE"
        }
    }

    if ($Extract) {
        $zipPath = Join-Path $VocalSetRoot "VocalSet1-2.zip"
        $extractMarker = Join-Path $VocalSetRoot ".extracted-vocalset-1-2"
        if (-not (Test-Path $extractMarker)) {
            Expand-Archive -LiteralPath $zipPath -DestinationPath $VocalSetRoot -Force
            New-Item -ItemType File -Path $extractMarker -Force | Out-Null
        }
    }

    $trainArgs = @(
        "-m", "gt_singer_grader.vocalset_quality",
        "--vocalset-root", $VocalSetRoot,
        "--output-dir", $OutputDir,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize"
    )
    if ($MaxRecords -gt 0) {
        $trainArgs += @("--max-records", "$MaxRecords")
    }

    if ($python -eq "py") {
        & $python -3 @trainArgs
    }
    else {
        & $python @trainArgs
    }
}
finally {
    Pop-Location
}
