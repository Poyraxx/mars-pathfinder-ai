param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskConfigPath = Join-Path $taskRoot '.local-run.json'
if (Test-Path -LiteralPath $taskConfigPath) {
    $taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
    $taskPython = $taskConfig.python
} else {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
Set-Location -LiteralPath $taskRoot
Write-Host "Mars Pathfinder AI: http://127.0.0.1:$Port"
& $taskPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
