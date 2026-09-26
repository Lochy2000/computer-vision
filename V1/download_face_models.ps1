$ErrorActionPreference = "Stop"

$modelsDirectory = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $modelsDirectory | Out-Null

$models = @(
    [pscustomobject]@{
        Name = "face_detection_yunet_2023mar.onnx"
        Uri = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        MinimumBytes = 200000
    },
    [pscustomobject]@{
        Name = "face_recognition_sface_2021dec.onnx"
        Uri = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        MinimumBytes = 30000000
    }
)

foreach ($model in $models) {
    $destination = Join-Path $modelsDirectory $model.Name
    if ((Test-Path -LiteralPath $destination) -and
        ((Get-Item -LiteralPath $destination).Length -ge $model.MinimumBytes)) {
        Write-Host "Already present: $($model.Name)"
        continue
    }
    $temporary = "$destination.download"
    Write-Host "Downloading $($model.Name)..."
    Invoke-WebRequest -Uri $model.Uri -OutFile $temporary
    if ((Get-Item -LiteralPath $temporary).Length -lt $model.MinimumBytes) {
        throw "Downloaded file is unexpectedly small: $($model.Name)"
    }
    Move-Item -Force -LiteralPath $temporary -Destination $destination
}

Write-Host "Face models are ready in $modelsDirectory"
