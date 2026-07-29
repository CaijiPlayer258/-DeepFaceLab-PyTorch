@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   准备 DST 数据
echo   1. 从视频提取帧
echo   2. 提取人脸
echo ============================================

set /p dst_video="请输入 DST 视频路径 (拖拽视频文件到此处): "

python main.py videoed extract-video ^
    --input-file "%dst_video%" ^
    --output-dir "workspace/data_dst"

python main.py extract ^
    --input-dir "workspace/data_dst" ^
    --output-dir "workspace/data_dst/aligned" ^
    --detector s3fd ^
    --face-type whole_face ^
    --image-size 512

python main.py sort ^
    --input-dir "workspace/data_dst/aligned" ^
    --by histogram

pause
