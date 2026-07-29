@echo off
chcp 65001 >nul
title SA-XSeg Benchmark (ONNX)
cd /d C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main

if not exist "workspace\model\SAXSeg\SAXSeg_256.onnx" (
    echo [ERROR] ONNX not found. Export with:
    echo   export_SAXSeg_onnx.bat
    pause
    exit /b
)

python workspace\HRN\bench_onnx_saxseg.py
pause
