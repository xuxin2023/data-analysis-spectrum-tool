$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryScript = Join-Path $projectRoot "fr_r_spectrum_tool_rebuild.py"
$iconPath = Join-Path $projectRoot "assets\app_icon.ico"
$guideSourcePath = Join-Path $projectRoot "quick_start.txt"
$distDir = Join-Path $projectRoot "dist"
$buildDir = Join-Path $projectRoot "build"
$specPath = Join-Path $projectRoot "data_analysis.spec"
$exePath = Join-Path $distDir "data_analysis.exe"
$guideDistPath = Join-Path $distDir "quick_start.txt"
$zipPath = Join-Path $distDir "data_analysis_win64.zip"

$exeZhName = [string]::Concat(([char]0x6570), ([char]0x636E), ([char]0x5206), ([char]0x6790), ".exe")
$guideZhName = [string]::Concat(([char]0x4F7F), ([char]0x7528), ([char]0x8BF4), ([char]0x660E), ".txt")
$zipZhName = [string]::Concat(([char]0x6570), ([char]0x636E), ([char]0x5206), ([char]0x6790), "_win64.zip")
$legacyBadGuideName = [string]::Concat(([char]0x6D63), ([char]0x8DE8), ([char]0x6564), ([char]0x7487), ([char]0x5B58), ([char]0x69D1), ".txt")
$legacyBadExeName = [string]::Concat(([char]0x93C1), ([char]0x7248), ([char]0x5D41), ([char]0x9352), ([char]0x55D8), ([char]0x703D), ".exe")

$exeZhPath = Join-Path $distDir $exeZhName
$guideZhDistPath = Join-Path $distDir $guideZhName
$zipZhPath = Join-Path $distDir $zipZhName
$legacyBadGuidePath = Join-Path $distDir $legacyBadGuideName
$legacyBadExePath = Join-Path $distDir $legacyBadExeName
$zipStageDir = Join-Path $distDir "_zip_stage"

if (-not (Test-Path $entryScript)) {
    throw "Entry script not found: $entryScript"
}

if (-not (Test-Path $iconPath)) {
    throw "Icon file not found: $iconPath"
}

if (-not (Test-Path $guideSourcePath)) {
    throw "Guide file not found: $guideSourcePath"
}

function Wait-FileReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [int]$RetryCount = 30,
        [int]$DelaySeconds = 2
    )

    for ($attempt = 0; $attempt -lt $RetryCount; $attempt++) {
        try {
            $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::None)
            $stream.Close()
            return
        } catch {
            Start-Sleep -Seconds $DelaySeconds
        }
    }

    throw "File stayed locked too long: $Path"
}

New-Item -ItemType Directory -Path $distDir -Force | Out-Null

Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
Remove-Item -Force $specPath -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $zipStageDir -ErrorAction SilentlyContinue
Remove-Item -Force $exePath -ErrorAction SilentlyContinue
Remove-Item -Force $guideDistPath -ErrorAction SilentlyContinue
Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
Remove-Item -Force $exeZhPath -ErrorAction SilentlyContinue
Remove-Item -Force $guideZhDistPath -ErrorAction SilentlyContinue
Remove-Item -Force $zipZhPath -ErrorAction SilentlyContinue
Remove-Item -Force $legacyBadGuidePath -ErrorAction SilentlyContinue
Remove-Item -Force $legacyBadExePath -ErrorAction SilentlyContinue

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "data_analysis" `
    --icon $iconPath `
    --add-data "$projectRoot\assets;assets" `
    --hidden-import "matplotlib.backends.backend_tkagg" `
    --hidden-import "matplotlib.backends._backend_tk" `
    $entryScript

if (-not (Test-Path $exePath)) {
    throw "EXE not found after build: $exePath"
}

Wait-FileReady -Path $exePath

Copy-Item -Path $guideSourcePath -Destination $guideDistPath -Force
Copy-Item -Path $exePath -Destination $exeZhPath -Force
Copy-Item -Path $guideSourcePath -Destination $guideZhDistPath -Force

New-Item -ItemType Directory -Path $zipStageDir -Force | Out-Null

$zipStageEnDir = Join-Path $zipStageDir "en"
$zipStageZhDir = Join-Path $zipStageDir "zh"

New-Item -ItemType Directory -Path $zipStageEnDir -Force | Out-Null
New-Item -ItemType Directory -Path $zipStageZhDir -Force | Out-Null

Copy-Item -Path $exePath -Destination (Join-Path $zipStageEnDir "data_analysis.exe") -Force
Copy-Item -Path $guideDistPath -Destination (Join-Path $zipStageEnDir "quick_start.txt") -Force
Copy-Item -Path $exeZhPath -Destination (Join-Path $zipStageZhDir $exeZhName) -Force
Copy-Item -Path $guideZhDistPath -Destination (Join-Path $zipStageZhDir $guideZhName) -Force

Compress-Archive -Path (Join-Path $zipStageEnDir "*") -DestinationPath $zipPath -Force
Compress-Archive -Path (Join-Path $zipStageZhDir "*") -DestinationPath $zipZhPath -Force

Remove-Item -Recurse -Force $zipStageDir -ErrorAction SilentlyContinue

Write-Host "EXE: $exePath"
Write-Host "GUIDE: $guideDistPath"
Write-Host "ZIP: $zipPath"
Write-Host "EXE_ZH: $exeZhPath"
Write-Host "GUIDE_ZH: $guideZhDistPath"
Write-Host "ZIP_ZH: $zipZhPath"
