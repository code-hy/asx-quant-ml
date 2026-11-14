# Weekly (or ad-hoc) full refresh of the platform, Phase 9.
# Usage:  powershell -File scripts\update.ps1 [-WithSensitivity]
# Restartable: if a stage fails, fix the cause and rerun the same command.

param(
    [switch]$WithSensitivity
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$args = @("run", "asx_quant", "update", "--mode", "incremental")
if ($WithSensitivity) { $args += "--with-sensitivity" }

Write-Host "Running: uv $($args -join ' ')"
& uv @args
exit $LASTEXITCODE
