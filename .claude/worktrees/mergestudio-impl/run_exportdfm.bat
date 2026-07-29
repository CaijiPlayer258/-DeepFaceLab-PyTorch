@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   导出 DeepFaceLive 模型
echo ============================================

echo 模型: SAEHD / AMP / Quick96
set /p model="模型 (默认 SAEHD): "
if "%model%"=="" set model=SAEHD

python main.py exportdfm ^
    --model %model% ^
    --model-dir "workspace/model"

echo OK
pause
