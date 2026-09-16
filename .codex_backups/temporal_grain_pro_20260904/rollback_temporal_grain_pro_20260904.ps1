$ErrorActionPreference = "Stop"

$backupRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeRoot = "D:\ComfyUI\ComfyUI\custom_nodes\IAMCCS-nodes"

Copy-Item -LiteralPath (Join-Path $backupRoot "temporal_film_grain.py") -Destination (Join-Path $nodeRoot "cine_nodes\temporal_film_grain.py") -Force
Copy-Item -LiteralPath (Join-Path $backupRoot "iamccs_temporal_film_grain_ui.js") -Destination (Join-Path $nodeRoot "web\iamccs_temporal_film_grain_ui.js") -Force

$proUi = Join-Path $nodeRoot "web\cine_nodes\temporal_film_grain_pro_ui.js"
if (Test-Path -LiteralPath $proUi) {
    Remove-Item -LiteralPath $proUi -Force
}

Write-Host "IAMCCS Temporal Grain PRO changes rolled back. Restart ComfyUI and refresh the browser."
