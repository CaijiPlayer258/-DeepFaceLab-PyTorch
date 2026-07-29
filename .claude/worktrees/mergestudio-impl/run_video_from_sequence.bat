@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   从合成帧生成视频
echo ============================================

set /p fps="FPS (默认 30): "
if "%fps%"=="" set fps=30
set /p bitrate="码率 Mbps (默认 15): "
if "%bitrate%"=="" set bitrate=15

python main.py videoed video-from-sequence ^
    --input-dir "workspace/data_dst/merged" ^
    --output-file "workspace/result.mp4" ^
    --reference-file "workspace/data_dst.*" ^
    --fps %fps% ^
    --bitrate %bitrate% ^
    --include-audio ^
    --ext png

echo OK
pause
