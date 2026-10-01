<#
Build the native offline Windows application and create a Desktop shortcut.
Run from PowerShell in this project directory:
    .\build_exe.ps1
#>

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

# No model is bundled: the checkpoint is chosen at runtime from the experiment
# folder the user selects, so the application ships without any .pt file.
$requiredAssets = @('hand_landmarker.task', 'face_detector.tflite', 'vosk_model')
foreach ($asset in $requiredAssets) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $asset))) {
        throw "Required offline asset is missing: $asset"
    }
}

$arguments = @(
    '--noconfirm', '--clean', '--windowed', '--onedir',
    '--name', 'SIH26174ActivityRecognition',
    '--collect-all', 'mediapipe',
    '--collect-all', 'cv2',
    '--collect-all', 'vosk',
    '--collect-all', 'ultralytics',
    '--collect-all', 'torch',
    '--hidden-import', 'argostranslate.translate',
    '--hidden-import', 'pyttsx3',
    '--hidden-import', 'sounddevice',
    '--hidden-import', 'bleak',
    '--add-data', "$projectRoot\hand_landmarker.task;.",
    '--add-data', "$projectRoot\face_detector.tflite;.",
    '--add-data', "$projectRoot\vosk_model;vosk_model",
    'desktop_app.py'
)

& python -m PyInstaller @arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$exe = Join-Path $projectRoot 'dist\SIH26174ActivityRecognition\SIH26174ActivityRecognition.exe'
& "$projectRoot\install_desktop_shortcut.ps1" -ExePath $exe
Write-Host "Built native application: $exe"
