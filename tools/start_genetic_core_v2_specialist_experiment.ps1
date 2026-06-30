param(
    [string]$RunDir,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Exchange = $env:CRYPTO_EXCHANGE,
    [string]$DataDir = "Retrodate",
    [int]$Population = 180,
    [int]$Generations = 12
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($Exchange)) {
    throw "Exchange is required; pass -Exchange MEXC or -Exchange BITGET"
}
$Exchange = $Exchange.ToUpperInvariant()

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $Root ("Results\neiro_genetics\" + $Exchange + "\genetic_core_v2_specialist_singleton_$stamp")
}
if (-not [System.IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $Root $RunDir
}

$global:LASTEXITCODE = $null
& (Join-Path $PSScriptRoot "start_genetics_v4_heavy_training.ps1") `
    -RunDir $RunDir `
    -Python $Python `
    -Exchange $Exchange `
    -DataDir $DataDir `
    -Population $Population `
    -Generations $Generations `
    -PositionStateFeatures
$scriptSucceeded = $?
if (-not $scriptSucceeded -or ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0)) {
    throw "start_genetics_v4_heavy_training.ps1 failed with exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $RunDir "launch_manifest.json"
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifest | Add-Member -NotePropertyName "experiment" -NotePropertyValue "GeneticCoreV2SpecialistToSingleton" -Force
    $manifest | Add-Member -NotePropertyName "specialist_regimes" -NotePropertyValue @("crash", "bearish", "neutral", "bullish") -Force
    $manifest | Add-Member -NotePropertyName "teacher_feature_agents" -NotePropertyValue @("LiveOIBreakout", "MomentumScalper", "ResearchValidatorAgent") -Force
    $manifest | Add-Member -NotePropertyName "live_policy" -NotePropertyValue "probation_only" -Force
    $manifest | Add-Member -NotePropertyName "uses_live_ensemble" -NotePropertyValue $false -Force
    $manifest | Add-Member -NotePropertyName "v2_report" -NotePropertyValue (Join-Path $RunDir "genetic_core_v2_experiment_summary.json") -Force
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
}

Write-Host "genetic_core_v2_run_dir=$RunDir"
