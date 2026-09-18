# Windows equivalent of run_all.sh, for the lab machines that log in as .\anees.
# Runs every measured part of HW2.5 on one card, then rebuilds figures and METRICS.md.
#
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Gpu 1
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Smoke
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Py py
#
# This script has not been run on Windows. If it misbehaves, run the five python
# commands by hand instead. They are listed in LAB_RUNBOOK.md and are what actually
# matter. Do not spend reserved time debugging a convenience wrapper.

param(
    [int]$Gpu = 0,
    [double]$Minutes = 20,
    [switch]$Smoke,
    [string]$Py = "python"
)

Set-Location $PSScriptRoot

$bArgs = @()
$cArgs = @()
$dArgs = @()

if ($Smoke) {
    $bArgs = @("--sizes", "1024", "2048", "--warmup", "2", "--iters", "5")
    $cArgs = @("--elements", "16777216", "--matmul-n", "2048", "--warmup", "2", "--iters", "5")
    $dArgs = @("--seq-lens", "512", "1024", "2048", "--warmup", "1", "--iters", "3", "--no-refine")
    $Minutes = 1
    Write-Host "SMOKE MODE: short sweeps and a $Minutes minute thermal run."
    Write-Host "These outputs are a chain check only. Delete them before the real run."
}

# A failing python script sets LASTEXITCODE but does not stop PowerShell on its own, so
# every step is checked explicitly. Otherwise Part A could fail and Part E would still
# run for 20 minutes.
function Invoke-Step {
    param([string]$Label, [string]$Script, [string[]]$Extra)

    Write-Host $Label
    & $Py (Join-Path "scripts" $Script) --index $Gpu @Extra
    if ($LASTEXITCODE -ne 0) {
        throw "$Script failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step "Part A: provenance" "part_a_provenance.py" @()
Invoke-Step "Part B: precision and achieved throughput" "part_b_precision.py" $bArgs
Invoke-Step "Part C: bandwidth bound against compute bound" "part_c_roofline.py" $cArgs
Invoke-Step "Part D: the cost of attention" "part_d_attention.py" $dArgs
Invoke-Step "Part E: $Minutes minutes of sustained load" "part_e_thermal.py" @("--minutes", "$Minutes")

Write-Host "Figures and METRICS.md"
& $Py (Join-Path "scripts" "make_figures.py")
if ($LASTEXITCODE -ne 0) { throw "make_figures.py failed with exit code $LASTEXITCODE" }
& $Py (Join-Path "scripts" "make_metrics.py")
if ($LASTEXITCODE -ne 0) { throw "make_metrics.py failed with exit code $LASTEXITCODE" }

if ($Smoke) {
    Write-Host ""
    Write-Host "Smoke run complete. If data, logs and figures filled in, the chain works."
    Write-Host "Now clear it before the real run:"
    Write-Host "  Remove-Item data\*.csv, data\*.json, logs\*.csv, figures\*.png, RUN_LOG.txt"
} else {
    Write-Host "Done. Review RUN_LOG.txt, METRICS.md and figures before committing."
}
