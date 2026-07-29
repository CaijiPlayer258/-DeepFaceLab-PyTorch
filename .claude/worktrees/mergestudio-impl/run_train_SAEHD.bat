@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   SAEHD 训练 (WebUI: http://localhost:6789)
echo ============================================

python main.py train ^
    --model SAEHD ^
    --training-data-src-dir "workspace/data_src/aligned" ^
    --training-data-dst-dir "workspace/data_dst/aligned" ^
    --model-dir "workspace/model"

pause
