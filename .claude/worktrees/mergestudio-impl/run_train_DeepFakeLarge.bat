@echo off
chcp 65001 >nul
cd /d "%~dp0"

title DeepFakeLarge Training

cls
echo ============================================================
echo   DeepFakeLarge Training (WebUI: http://localhost:6789)
echo ============================================================
echo.

python main.py train ^
    --model DeepFakeLarge ^
    --training-data-src-dir "workspace/data_src/aligned" ^
    --training-data-dst-dir "workspace/data_dst/aligned" ^
    --model-dir "workspace/model/DeepFakeLarge"

pause
