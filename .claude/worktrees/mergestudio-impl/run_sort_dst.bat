@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   排序 DST 人脸（按直方图相似度）
echo ============================================

python main.py sort ^
    --input-dir "workspace/data_dst/aligned" ^
    --by histogram

echo OK
pause
