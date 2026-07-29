@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   WebUI 独立监控 :6789
echo   配合训练使用（训练内置 :6789）
echo ============================================

python WebUI\page_trainer.py --model-dir "workspace/model" --port 8081

pause
