@echo off
REM 启动独立文件监控进程
REM 用法: runFileMonitor.bat [src_path] [dst_path] [interval]

set SRC_PATH=%1
set DST_PATH=%2
set INTERVAL=%3

if "%SRC_PATH%"=="" set SRC_PATH=workspace\data_src\aligned
if "%DST_PATH%"=="" set DST_PATH=workspace\data_dst\aligned
if "%INTERVAL%"=="" set INTERVAL=1.0

echo ========================================
echo 启动独立文件监控进程
echo ========================================
echo src: %SRC_PATH%
echo dst: %DST_PATH%
echo 检查间隔: %INTERVAL%s
echo ========================================
echo.

python samplelib\FileMonitorProcess.py --src "%SRC_PATH%" --dst "%DST_PATH%" --interval %INTERVAL%

pause
