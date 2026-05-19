param(
    [string]$Checkpoint = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Device = "auto",
    [switch]$NoBrowser
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $repoRoot) ".venvs\nanopitch\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

$args = @(
    "-m", "gt_singer_grader.demo",
    "--host", $BindHost,
    "--port", $Port,
    "--device", $Device
)

if ($Checkpoint) {
    $args += @("--checkpoint", $Checkpoint)
}

if (-not $NoBrowser) {
    $args += "--open-browser"
}

Push-Location $repoRoot
try {
    & $python @args
}
finally {
    Pop-Location
}
