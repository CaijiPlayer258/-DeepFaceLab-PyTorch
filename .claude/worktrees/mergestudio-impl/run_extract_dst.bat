@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   提取 DST 人脸
echo ============================================

python main.py extract ^
    --input-dir "workspace/data_dst" ^
    --output-dir "workspace/data_dst/aligned" ^
    --detector s3fd ^
    --face-type whole_face ^
    --image-size 512

echo OK
pause
