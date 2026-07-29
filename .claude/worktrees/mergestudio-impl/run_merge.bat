@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   合成 (Merge)
echo ============================================

echo 模型: SAEHD / AMP / Quick96 / XSeg
set /p model="模型 (默认 SAEHD): "
if "%model%"=="" set model=SAEHD

python main.py merge ^
    --model %model% ^
    --model-dir "workspace/model" ^
    --input-dir "workspace/data_dst" ^
    --output-dir "workspace/data_dst/merged" ^
    --output-mask-dir "workspace/data_dst/merged_mask"

echo OK
pause
