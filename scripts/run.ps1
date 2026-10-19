$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Python environment missing. Follow README.md to install dependencies.'
}
if (-not (Test-Path -LiteralPath 'backend\.env')) {
    Copy-Item -LiteralPath 'backend\.env.example' -Destination 'backend\.env'
}
if (-not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
    npm.cmd --prefix frontend run build -- --configLoader runner
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
& '.\.venv\Scripts\python.exe' -m scripts.serve
if ($LASTEXITCODE -ne 0) {
    throw "The backend exited with code $LASTEXITCODE. Review the startup error above."
}
