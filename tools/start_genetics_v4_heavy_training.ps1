param(
    [string]$RunDir,
    [string]$TrainStartDate = "2022-01-01",
    [string]$TrainEndDate = "2023-12-31",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Exchange = $env:CRYPTO_EXCHANGE,
    [string]$DataDir = "Retrodate",
    [int]$Population = 900,
    [int]$Generations = 100,
    [switch]$PositionStateFeatures,
    [switch]$NoPostTraining
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($Exchange)) {
    throw "Exchange is required for genetics heavy training; pass -Exchange MEXC or -Exchange BITGET"
}
$Exchange = $Exchange.ToUpperInvariant()

if ($TrainStartDate -ne "2022-01-01" -or $TrainEndDate -ne "2023-12-31") {
    throw "GeneticCore training train window must be exactly 2022-01-01..2023-12-31"
}

$ProvenBcSeedAgents = @("LiveOIBreakout", "MomentumScalper", "ResearchValidatorAgent")
$CostProfile = switch ($Exchange) {
    "MEXC" {
        @{
            train_fee = "0.0004"
            train_futures_fee = "0.0002"
            train_funding_rate = "0.0001"
            train_spread = "0.0002"
            train_slippage = "0.0001"
            train_min_notional_usd = "5.0"
            train_quantity_precision_step = "0.001"
        }
    }
    "BITGET" {
        @{
            train_fee = "0.0006"
            train_futures_fee = "0.0004"
            train_funding_rate = "0.0001"
            train_spread = "0.0002"
            train_slippage = "0.00015"
            train_min_notional_usd = "5.0"
            train_quantity_precision_step = "0.001"
        }
    }
    default {
        throw "Unsupported exchange for GeneticCore cost profile: $Exchange"
    }
}

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $Root ("Results\neiro_genetics\" + $Exchange.ToUpperInvariant() + "\heavy_evolution_v4_$stamp")
}
if (-not [System.IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $Root $RunDir
}
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path $Root $Python
}
if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $Root $DataDir
}

$AgentsDir = Join-Path $RunDir "agents"
New-Item -ItemType Directory -Path $AgentsDir -Force | Out-Null

$baseSettings = Join-Path $Root "Genetics_DL_Agents\settings_genetic.txt"
$runSettings = Join-Path $RunDir "settings_genetic_train.txt"
Copy-Item -Path $baseSettings -Destination $runSettings -Force

$positionState = if ($PositionStateFeatures) { "on" } else { "off" }
$overrides = @(
    "",
    "# v4 experiment overrides",
    "pop_size = $Population",
    "n_generations = $Generations",
    "cpu_batch_evaluator = on",
    "position_state_features_enabled = $positionState",
    "continue_training = off",
    "load_best_genome = off",
    "load_extra_genomes = off",
    "load_regime_genomes = off",
    "load_island_genomes = off",
    "n_workers = 1",
    "bc_enabled = on",
    "bc_auto_discovery = off",
    "bc_min_active_ratio = 0.005",
    ("agent_seed_list = {0}" -f ($ProvenBcSeedAgents -join ",")),
    "exchange_cost_profile = $Exchange",
    ("train_fee = {0}" -f $CostProfile["train_fee"]),
    ("train_futures_fee = {0}" -f $CostProfile["train_futures_fee"]),
    ("train_funding_rate = {0}" -f $CostProfile["train_funding_rate"]),
    ("train_spread = {0}" -f $CostProfile["train_spread"]),
    ("train_slippage = {0}" -f $CostProfile["train_slippage"]),
    ("train_min_notional_usd = {0}" -f $CostProfile["train_min_notional_usd"]),
    ("train_quantity_precision_step = {0}" -f $CostProfile["train_quantity_precision_step"]),
    "open_confidence_min = 0.45",
    "open_confidence_penalty_w = 25.0",
    "open_logit_margin_min = 0.12",
    "open_logit_margin_penalty_w = 35.0",
    "robust_concentration_max_pct = 20.0",
    "robust_concentration_penalty_w = 0.75",
    "robust_positive_period_target = 0.60",
    "robust_positive_period_penalty_w = 7.5",
    "fitness_outlier_concentration_max_pct = 12.5",
    "fitness_outlier_concentration_penalty_w = 1.20",
    "fitness_direction_bias_max_abs = 0.50",
    "fitness_direction_bias_penalty_w = 20.0",
    "fitness_max_zero_period_rate = 0.25",
    "fitness_zero_period_penalty_w = 25.0",
    "fitness_persistent_direction_bias_max_abs = 0.25",
    "fitness_persistent_direction_bias_penalty_w = 30.0",
    "fitness_hard_gate_sentinel = -100000000.0",
    "fitness_hard_max_zero_period_rate = 0.50",
    "fitness_active_period_min_abs_ret = 0.10",
    "fitness_hard_min_active_period_rate = 0.35",
    "fitness_hard_min_mean_ret = 0.05",
    "fitness_micro_positive_max_ret = 0.10",
    "fitness_micro_positive_max_rate = 0.75",
    "fitness_micro_positive_mean_ret_ceiling = 0.20",
    "fitness_wfa_fold_count = 4",
    "fitness_wfa_min_fold_mean_ret = 0.05",
    "fitness_wfa_fold_penalty_w = 45.0",
    "fitness_wfa_fold_dispersion_penalty_w = 4.0",
    "fitness_wfa_hard_min_fold_mean_ret = -1000000000.0",
    "fitness_wfa_min_positive_fold_rate = 0.75",
    "fitness_wfa_positive_fold_penalty_w = 25.0",
    "train_start_date = $TrainStartDate",
    "train_end_date = $TrainEndDate"
)
Add-Content -Path $runSettings -Value $overrides -Encoding UTF8

$baselineSource = Join-Path $Root "Genetics_DL_Agents\Agents\genetics\best_genome.npy"
$baselineCopy = Join-Path $RunDir "baseline_control_best_genome.npy"
if (Test-Path -LiteralPath $baselineSource) {
    Copy-Item -Path $baselineSource -Destination $baselineCopy -Force
}

@{
    run_dir = $RunDir
    agents_dir = $AgentsDir
    exchange = $Exchange
    data_dir = $DataDir
    train_start_date = $TrainStartDate
    train_end_date = $TrainEndDate
    population = $Population
    generations = $Generations
    position_state_features_enabled = $positionState
    walk_forward_contract = @{
        train = @($TrainStartDate, $TrainEndDate)
        validation = @("2024-01-01", "2024-12-31")
        oos = @("2025-01-01", "2025-12-31")
        sanity = @("2026-01-01", "2026-06-30")
    }
    bc_seed_agents = $ProvenBcSeedAgents
    exchange_cost_profile = $CostProfile
    settings_file = $runSettings
    baseline_genome = if (Test-Path -LiteralPath $baselineCopy) { $baselineCopy } else { $null }
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $RunDir "launch_manifest.json") -Encoding UTF8

$stdout = Join-Path $RunDir "crypto_genetics_stdout.log"
$stderr = Join-Path $RunDir "crypto_genetics_stderr.log"

function Quote-CmdArg([string]$Value) {
    '"' + ($Value -replace '"', '\"') + '"'
}

function Env-CmdPrefix([hashtable]$EnvOverrides) {
    if ($null -eq $EnvOverrides -or $EnvOverrides.Count -eq 0) {
        return ""
    }
    $parts = @()
    foreach ($key in $EnvOverrides.Keys) {
        $value = [string]($EnvOverrides[$key])
        $parts += ('set "' + $key + '=' + ($value -replace '"', '\"') + '"')
    }
    return (($parts -join " && ") + " && ")
}

function Start-CmdRedirectedProcess {
    param(
        [string]$CommandLine,
        [hashtable]$EnvOverrides = @{}
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "$env:SystemRoot\System32\cmd.exe"
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.Arguments = '/d /s /c "' + $CommandLine + '"'
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    return $proc
}

$trainEnv = @{
    GENETICS_SETTINGS_FILE = $runSettings
    GENETICS_TRAIN_START_DATE = $TrainStartDate
    GENETICS_TRAIN_END_DATE = $TrainEndDate
    GENETICS_AGENTS_DIR = $AgentsDir
    GENETICS_DATA_DIR = $DataDir
    CRYPTO_EXCHANGE = $Exchange
    GENETICS_FORCE_FLUSH = "1"
    PYTHONUNBUFFERED = "1"
    PYTHONUTF8 = "1"
}
$trainCommand = (
    (Env-CmdPrefix $trainEnv) +
    (Quote-CmdArg $Python) + " -u " +
    (Quote-CmdArg "Genetics_DL_Agents\crypto_genetics.py") +
    " 1> " + (Quote-CmdArg $stdout) +
    " 2> " + (Quote-CmdArg $stderr)
)
$trainProc = Start-CmdRedirectedProcess -CommandLine $trainCommand -EnvOverrides $trainEnv

Set-Content -Path (Join-Path $RunDir "heavy_training_pid.txt") -Value $trainProc.Id -Encoding ASCII

if (-not $NoPostTraining) {
    $postStdout = Join-Path $RunDir "post_training_runner_stdout.log"
    $postStderr = Join-Path $RunDir "post_training_runner_stderr.log"
    $postArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "tools\run_genetics_v4_post_training.ps1",
        "-WaitPid", [string]$trainProc.Id,
        "-RunDir", $RunDir,
        "-AgentsDir", $AgentsDir,
        "-Exchange", $Exchange,
        "-DataDir", $DataDir
    )
    if (Test-Path -LiteralPath $baselineCopy) {
        $postArgs += @("-BaselineGenome", $baselineCopy)
    }
    $postCommand = (
        (Quote-CmdArg "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe") +
        " " +
        (($postArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") +
        " 1> " + (Quote-CmdArg $postStdout) +
        " 2> " + (Quote-CmdArg $postStderr)
    )
    $postProc = Start-CmdRedirectedProcess -CommandLine $postCommand -EnvOverrides @{}
    Set-Content -Path (Join-Path $RunDir "post_training_pid.txt") -Value $postProc.Id -Encoding ASCII
}

Write-Host "started heavy training pid=$($trainProc.Id)"
Write-Host "run_dir=$RunDir"
Write-Host "agents_dir=$AgentsDir"
