[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeRawSweeps,
    [switch]$IncludeLegacyArtifacts,
    [switch]$IncludeObsoletePlayerRetro,
    [int]$LogRetentionDays = 14
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ProjectPrefix = $ProjectRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
$Now = Get-Date

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $candidate = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RelativePath))
    if (-not $candidate.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside project root: $candidate"
    }
    return $candidate
}

function Get-PathBytes {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)

    if (-not (Test-Path -LiteralPath $LiteralPath)) {
        return [int64]0
    }
    $item = Get-Item -LiteralPath $LiteralPath -Force
    if (-not $item.PSIsContainer) {
        return [int64]$item.Length
    }
    $sum = Get-ChildItem -LiteralPath $LiteralPath -Recurse -Force -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum
    if ($null -eq $sum.Sum) {
        return [int64]0
    }
    return [int64]$sum.Sum
}

function Add-CleanupTarget {
    param(
        [System.Collections.Generic.List[object]]$Targets,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup target outside project root: $resolved"
    }
    $Targets.Add([pscustomobject]@{
        Path = $resolved
        Reason = $Reason
        Bytes = Get-PathBytes -LiteralPath $resolved
    })
}

$targets = [System.Collections.Generic.List[object]]::new()

foreach ($relative in @(".pytest_cache", ".tmp_retro_whatif_tests", "__pycache__")) {
    Add-CleanupTarget -Targets $targets -Path (Resolve-ProjectPath $relative) -Reason "runtime/test cache"
}

foreach ($cacheRoot in @("src", "tests", "tools", "Genetics_DL_Agents")) {
    $resolvedCacheRoot = Resolve-ProjectPath $cacheRoot
    if (Test-Path -LiteralPath $resolvedCacheRoot) {
        Get-ChildItem -LiteralPath $resolvedCacheRoot -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            ForEach-Object {
                Add-CleanupTarget -Targets $targets -Path $_.FullName -Reason "Python bytecode cache"
            }
    }
}

$logRoot = Resolve-ProjectPath "logs"
if (Test-Path -LiteralPath $logRoot) {
    $cutoff = $Now.AddDays(-[Math]::Max(0, $LogRetentionDays))
    Get-ChildItem -LiteralPath $logRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
            Add-CleanupTarget -Targets $targets -Path $_.FullName -Reason "runtime log older than $LogRetentionDays days"
        }
}

if ($IncludeRawSweeps) {
    $obsoleteSweepRoots = @(
        "Results\BitgetPolicyCandidates",
        "Results\BitgetHypothesisSweep",
        "Results\BitgetPolicyCanary",
        "Results\Panteon3PreLiveMatrix",
        "Results\Panteon3PreLiveMatrixBitgetFocus_20260629",
        "Results\BitgetCandidateRefill",
        "Results\BitgetRangeMatrix"
    )
    foreach ($relative in $obsoleteSweepRoots) {
        Add-CleanupTarget -Targets $targets -Path (Resolve-ProjectPath $relative) -Reason "completed raw sweep; compact reports/evidence retained"
    }
}

if ($IncludeObsoletePlayerRetro) {
    # These runs were produced while correcting attribution, position lifetime,
    # and full-market player invocation.  Their conclusions are superseded by
    # the hierarchical multi and ResearchValidator singleton evidence.
    $obsoletePlayerRetroRoots = @(
        "Results\PlayerEfficiencyBaselines",
        "Results\PlayerEfficiencyWalkForward",
        "Results\PlayerEfficiencyWalkForwardEntryRegime",
        "Results\PlayerEfficiencyWalkForwardLifecycle",
        "Results\PlayerEfficiencyWalkForwardCorrected",
        "Results\PlayerEfficiencyWalkForwardLifecycleMin5",
        "Results\PlayerEfficiencyWalkForwardCorrectedMin5",
        "Results\PlayerEfficiencySmoke",
        "Results\PlayerEfficiencyLifecycleSmoke",
        "Results\PlayerEfficiencyFreshShadowSmoke",
        "Results\PlayerEfficiencyHierarchicalSmoke"
    )
    foreach ($relative in $obsoletePlayerRetroRoots) {
        Add-CleanupTarget -Targets $targets -Path (Resolve-ProjectPath $relative) -Reason "superseded or incomplete player retro run"
    }
}

if ($IncludeLegacyArtifacts) {
    $legacyRoots = @(
        "Results\MEXC",
        "Results\Panteon3AgentDiagnostics",
        "Results\Panteon3BitgetCanaryAfterContextDeny",
        "Results\Panteon3PaperCanary",
        "Results\Panteon3PaperCanaryAfterRangePromotion",
        "Results\Panteon3PaperCanaryDiagnostics",
        "Results\Panteon3PaperCanaryFixes",
        "Results\Panteon3PaperCanaryNext",
        "Results\Panteon3PaperCanaryNoEvidence",
        "Results\Panteon3PaperCanaryP0",
        "Results\Panteon3PaperCanaryP0SeedPrior",
        "Results\Panteon3PaperCanaryP0SeedPriorMinNotional",
        "Results\Panteon3PaperCanaryP0ShutdownFlatten",
        "Results\Panteon3PaperCanaryP0TimerReset",
        "Results\Panteon3PaperCanaryP0VolumeAlign",
        "Results\Panteon3SingleComponentCanary",
        "Results\Panteon3SingleComponentCanary_isolated",
        "Results\Panteon3SingleComponentCanaryP0",
        "Results\Panteon3SingleComponentCanaryP0BestComponentNext",
        "Results\Panteon3SingleComponentCanaryP0BothNext",
        "Results\Panteon3SingleComponentCanaryP0Diag",
        "Results\Panteon3SingleComponentCanaryP0Fixed",
        "Results\Panteon3SingleComponentCanaryP0Long70",
        "Results\Panteon3SingleComponentCanaryP0Long70BitgetMinHalf",
        "Results\Panteon3SingleComponentCanaryP0MinNotional",
        "Results\Panteon3SingleComponentCanaryP0VolCompressNext",
        "Results\Panteon3SingleComponentCanarySmoke",
        "Results\DebugSingleLiveOIBreakoutCurrentActionable2",
        "Results\DebugSingleLiveOIBreakoutFixed",
        "Results\DebugSingleLiveOIBreakoutPromotionDerived",
        "Results\tmp_mexc_download_probe",
        "Reports\PreLive",
        "Reports\BitgetRangeBreakout",
        "Reports\Panteon3PreLiveMatrix",
        "Reports\Panteon3Diagnostics",
        "Reports\Panteon3Canary",
        "Reports\PolicyReplayV1",
        "Reports\BitgetPolicyCandidates",
        "Reports\BitgetHypothesisSweep",
        "Reports\BitgetCandidateRefill",
        "Reports\BitgetRangeMatrix",
        "Reports\Panteon3PaperCanary",
        "Reports\Panteon3PaperCanaryFixes",
        "Reports\Panteon3PaperCanaryNoEvidence",
        "Reports\Panteon3Compare",
        "Reports\PanteonFlashFinalTradingReadiness_20260523",
        "Reports\PanteonFlashPreLive_20260524",
        "Reports\PanteonFlashFiveYearTimeline_20260525"
    )
    foreach ($relative in $legacyRoots) {
        Add-CleanupTarget -Targets $targets -Path (Resolve-ProjectPath $relative) -Reason "legacy policy/Flash/Panteon3 artifact"
    }

    $trainingRoot = Resolve-ProjectPath "Results\neiro_genetics"
    if (Test-Path -LiteralPath $trainingRoot) {
        $rawTrainingNames = @(
            "select_router_fitness_v4.log",
            "selection_router_fitness_v4.json",
            "selection_router_fitness_v4_guard.json",
            "post_training_runner_stdout.log",
            "population_state.pkl",
            "genetic_core_v2_experiment_summary.json"
        )
        Get-ChildItem -LiteralPath $trainingRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object { $rawTrainingNames -contains $_.Name } |
            ForEach-Object {
                Add-CleanupTarget -Targets $targets -Path $_.FullName -Reason "raw MEXC training trace; genomes and compact validation retained"
            }
    }
}

$targets = @($targets | Sort-Object Path -Unique)
$measuredBytes = ($targets | Measure-Object -Property Bytes -Sum).Sum
$totalBytes = if ($null -eq $measuredBytes) { [int64]0 } else { [int64]$measuredBytes }
$mode = if ($Apply) { "APPLY" } else { "DRY-RUN" }
Write-Output "Project cleanup mode: $mode"
Write-Output "Project root: $ProjectRoot"
Write-Output "Targets: $($targets.Count); reclaimable: $([Math]::Round($totalBytes / 1GB, 3)) GiB"
$targets | Select-Object Path, Reason, @{Name = "MiB"; Expression = { [Math]::Round($_.Bytes / 1MB, 2) }} |
    Format-Table -AutoSize

if (-not $Apply) {
    Write-Output "No files were removed. Re-run with -Apply after reviewing the paths."
    exit 0
}

foreach ($target in $targets) {
    $resolved = [System.IO.Path]::GetFullPath([string]$target.Path)
    if (-not $resolved.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup target outside project root: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

Write-Output "Cleanup complete; reclaimed approximately $([Math]::Round($totalBytes / 1GB, 3)) GiB."
