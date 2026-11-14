# Start the Streamlit research dashboard (local-first, no server exposure).
# Usage:  powershell -File scripts\dashboard.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Starting the ASX quant dashboard on http://localhost:8501 ..."
& uv run streamlit run dashboard/Home.py --server.port 8501 --server.headless $false
