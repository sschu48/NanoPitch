param(
    [ValidateSet("gt_singer_only", "gt_singer_vocalset")]
    [string]$ModelProfile = "gt_singer_only",
    [string]$Checkpoint = "",
    [string]$QualityCheckpoint = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Device = "auto",
    [switch]$DisableQuality,
    [switch]$NoBrowser
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

    throw "No Python runtime was found. Activate a venv or install Python, then rerun the launcher."
}

$python = Resolve-PythonCommand

$args = @(
    "-m", "gt_singer_grader.demo",
    "--model-profile", $ModelProfile,
    "--host", $BindHost,
    "--port", $Port,
    "--device", $Device
)

if ($Checkpoint) {
    $args += @("--checkpoint", $Checkpoint)
}

if ($QualityCheckpoint) {
    $args += @("--quality-checkpoint", $QualityCheckpoint)
}

if ($DisableQuality) {
    $args += "--disable-quality"
}

if (-not $NoBrowser) {
    $args += "--open-browser"
}

Push-Location $repoRoot
try {
    if ($python -eq "py") {
        & $python -3 @args
    }
    else {
        & $python @args
    }
}
finally {
    Pop-Location
}
