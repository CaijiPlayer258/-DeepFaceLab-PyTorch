@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   从视频提取帧 -- SRC
echo ============================================

set /p video_path="SRC 视频: "

python main.py videoed extract-video ^
    --input-file "%video_path%" ^
    --output-dir "workspace/data_src"

echo OK
pause
