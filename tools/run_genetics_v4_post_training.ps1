param(
    [int]$WaitPid = 0,
    [string]$RunDir,
    [string]$AgentsDir = "Genetics_DL_Agents\Agents\genetics",
    [string]$BaselineGenome,
    [string]$Exchange = $env:CRYPTO_EXCHANGE,
    [string]$DataDir = "Retrodate",
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if ([string]::IsNullOrWhiteSpace($Exchange)) {
        throw "Exchange is required for genetics post-training; pass -Exchange MEXC or -Exchange BITGET"
    }
    $RunDir = Join-Path $Root ("Results\neiro_genetics\" + $Exchange.ToUpperInvariant() + "\post_training_v4_$stamp")
}
if (-not [System.IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $Root $RunDir
}
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

if (-not [System.IO.Path]::IsPathRooted($AgentsDir)) {
    $AgentsDir = Join-Path $Root $AgentsDir
}
if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path $Root $Python
}
if (-not [string]::IsNullOrWhiteSpace($BaselineGenome) -and -not [System.IO.Path]::IsPathRooted($BaselineGenome)) {
    $BaselineGenome = Join-Path $Root $BaselineGenome
}

function Write-Step($Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $RunDir "post_training_v4.log") -Value $line
}

function Invoke-LoggedPython($Name, [string[]]$ArgsList) {
    $log = Join-Path $RunDir "$Name.log"
    Write-Step "running $Name"
    Write-Step ("command: " + $Python + " " + ($ArgsList -join " "))
    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if ($WaitPid -gt 0) {
    Write-Step "waiting for heavy training pid=$WaitPid"
    Wait-Process -Id $WaitPid
    Write-Step "heavy training pid=$WaitPid completed"
}

$genomePaths = New-Object System.Collections.Generic.List[string]
if (-not [string]::IsNullOrWhiteSpace($BaselineGenome) -and (Test-Path -LiteralPath $BaselineGenome)) {
    $genomePaths.Add((Resolve-Path -LiteralPath $BaselineGenome).Path)
}

$patterns = @(
    "best_genome.npy",
    "archive_rank*.npy",
    "best_island_*.npy",
    "best_genome_regime_*.npy"
)
foreach ($pattern in $patterns) {
    Get-ChildItem -Path $AgentsDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Sort-Object Name |
        ForEach-Object { $genomePaths.Add($_.FullName) }
}

$uniqueGenomes = @()
$seen = @{}
foreach ($path in $genomePaths) {
    if (-not $seen.ContainsKey($path)) {
        $seen[$path] = $true
        $uniqueGenomes += $path
    }
}
if ($uniqueGenomes.Count -lt 2) {
    throw "Need at least baseline and one candidate genome for v4 selection"
}

$manifest = [ordered]@{
    generated_at = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    exchange = $Exchange.ToUpperInvariant()
    wait_pid = $WaitPid
    run_dir = $RunDir
    agents_dir = $AgentsDir
    baseline_genome = $BaselineGenome
    genome_count = $uniqueGenomes.Count
    genomes = $uniqueGenomes
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $RunDir "post_training_genomes.json") -Encoding UTF8

$genomeArgs = @()
foreach ($path in $uniqueGenomes) {
    $genomeArgs += @("--genome", $path)
}

$trainReport = Join-Path $RunDir "contract_train_2022_2023.json"
$validationReport = Join-Path $RunDir "contract_validation_2024.json"
$oosReport = Join-Path $RunDir "contract_oos_2025.json"
$finalReport = Join-Path $RunDir "contract_final_sanity_2026_h1.json"

$evalTrainArgs = @(
    "tools\evaluate_genetics_contract.py"
) + $genomeArgs + @(
    "--out", $trainReport,
    "--start-date", "2022-01-01",
    "--end-date", "2023-12-31",
    "--data-dir", $DataDir,
    "--exchange", $Exchange,
    "--walk-forward",
    "--train-years", "2",
    "--validation-years", "1",
    "--final-test-year", "2024",
    "--embargo-months", "1"
)
Invoke-LoggedPython "eval_train_2022_2023" $evalTrainArgs

$evalValidationArgs = @(
    "tools\evaluate_genetics_contract.py"
) + $genomeArgs + @(
    "--out", $validationReport,
    "--start-date", "2024-01-01",
    "--end-date", "2024-12-31",
    "--data-dir", $DataDir,
    "--exchange", $Exchange
)
Invoke-LoggedPython "eval_validation_2024" $evalValidationArgs

$evalOosArgs = @(
    "tools\evaluate_genetics_contract.py"
) + $genomeArgs + @(
    "--out", $oosReport,
    "--start-date", "2025-01-01",
    "--end-date", "2025-12-31",
    "--data-dir", $DataDir,
    "--exchange", $Exchange
)
Invoke-LoggedPython "eval_oos_2025" $evalOosArgs

$evalFinalArgs = @(
    "tools\evaluate_genetics_contract.py"
) + $genomeArgs + @(
    "--out", $finalReport,
    "--start-date", "2026-01-01",
    "--end-date", "2026-06-30",
    "--data-dir", $DataDir,
    "--exchange", $Exchange
)
Invoke-LoggedPython "eval_final_sanity_2026_h1" $evalFinalArgs

Invoke-LoggedPython "select_single_fitness_v4" @(
    "tools\select_genetics_candidate.py",
    "--train-report", $trainReport,
    "--validation-report", $validationReport,
    "--final-report", $oosReport,
    "--final-report", $finalReport,
    "--out", (Join-Path $RunDir "selection_single_fitness_v4.json"),
    "--exchange", $Exchange,
    "--use-fitness-v4-robust",
    "--min-validation-mean-delta", "0.0",
    "--min-validation-min-ret-delta", "0.0",
    "--min-positive-period-pct", "50.0",
    "--max-turnover-rate", "0.20",
    "--max-saturation-rate", "0.15",
    "--max-invalid-open-pressure", "0.25"
)

Invoke-LoggedPython "select_router_fitness_v4" @(
    "tools\select_genetics_candidate.py",
    "--train-report", $trainReport,
    "--validation-report", $validationReport,
    "--regime-router",
    "--allowed-candidate-regime", "crash",
    "--allowed-candidate-regime", "bearish",
    "--allowed-candidate-regime", "neutral",
    "--allowed-candidate-regime", "bullish",
    "--final-report", $oosReport,
    "--final-report", $finalReport,
    "--out", (Join-Path $RunDir "selection_router_fitness_v4.json"),
    "--exchange", $Exchange,
    "--min-validation-mean-delta", "0.0",
    "--min-validation-min-ret-delta", "0.0",
    "--min-positive-period-pct", "50.0",
    "--max-turnover-rate", "0.20",
    "--max-saturation-rate", "0.15",
    "--max-invalid-open-pressure", "0.25",
    "--min-validation-periods", "6",
    "--min-validation-regime-periods", "1",
    "--min-oos-regime-periods", "1"
)

Write-Step "post-training v4 pipeline completed"
