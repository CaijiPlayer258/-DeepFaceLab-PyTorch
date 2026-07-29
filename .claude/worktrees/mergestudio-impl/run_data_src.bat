@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   准备 SRC 数据
echo   1. 从视频提取帧
echo   2. 提取人脸
echo ============================================

set /p src_video="请输入 SRC 视频路径 (拖拽视频文件到此处): "

python main.py videoed extract-video ^
    --input-file "%src_video%" ^
    --output-dir "workspace/data_src"

python main.py extract ^
    --input-dir "workspace/data_src" ^
    --output-dir "workspace/data_src/aligned" ^
    --detector s3fd ^
    --face-type whole_face ^
    --image-size 512

python main.py sort ^
    --input-dir "workspace/data_src/aligned" ^
    --by histogram

pause
