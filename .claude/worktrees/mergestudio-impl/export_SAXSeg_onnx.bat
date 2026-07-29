@echo off
chcp 65001 >nul
title Export SA-XSeg to ONNX
cd /d C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main

set PTH=workspace\model\SAXSeg\SAXSeg_256.pth
if not exist "%PTH%" (
    echo [ERROR] Model not found: %PTH%
    pause
    exit /b
)

echo Exporting SA-XSeg to ONNX...
echo Model: %PTH%
echo.

python workspace\HRN\export_saxseg_onnx.py --model %PTH%
pause
