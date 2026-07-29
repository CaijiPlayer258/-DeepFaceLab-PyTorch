@echo off
chcp 65001 >nul
title XSegLite Training
cd /d C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main

cls
echo ============================================================
echo        XSegLite Training  (Lightweight CNN XSeg)
echo ============================================================
echo.
echo Pure Conv3x3 U-Net, 4-stage  ^|  ECA + Dice loss
echo 4.75M params  ^|  3.0ms PyTorch / 2.5ms ONNX (RTX 3080)  ^|  18MB
echo.

set MODEL_DIR=workspace\model\XSegLite
set SRC_DIR=workspace\data_src\aligned
set DST_DIR=workspace\data_src\aligned

if not exist "%SRC_DIR%" (
    echo [ERROR] SRC faceset not found: %SRC_DIR%
    pause
    exit /b
)
if not exist "%DST_DIR%" (
    echo [ERROR] DST faceset not found: %DST_DIR%
    pause
    exit /b
)

python main.py train --model XSegLite --model-dir %MODEL_DIR% --training-data-src-dir %SRC_DIR% --training-data-dst-dir %DST_DIR%

pause
