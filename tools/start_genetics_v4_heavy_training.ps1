param(
    [string]$RunDir,
    [string]$TrainStartDate = "2022-01-01",
    [string]$TrainEndDate = "2023-12-31",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$Population = 900,
    [int]$Generations = 100,
    [switch]$PositionStateFeatures,
    [switch]$NoPostTraining
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $Root "Results\neiro_genetics\heavy_evolution_v4_$stamp"
}
if (-not [System.IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $Root $RunDir
}
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path $Root $Python
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
    "agent_seed_list = Panteon_Flash",
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
    train_start_date = $TrainStartDate
    train_end_date = $TrainEndDate
    population = $Population
    generations = $Generations
    position_state_features_enabled = $positionState
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
        "-AgentsDir", $AgentsDir
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
