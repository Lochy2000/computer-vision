$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error "The v3 virtual environment was not found. Run: py -m venv .venv"
    exit 1
}

& $pythonPath -m drone_detection.cli @args
exit $LASTEXITCODE
