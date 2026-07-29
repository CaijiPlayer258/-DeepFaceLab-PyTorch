@echo off
CHCP 65001 >nul
title FFmpeg 更新工具 — 完整版（含 HDR 色调映射）

echo ============================================================
echo       FFmpeg 完整版更新工具
echo       下载带 libplacebo 的版本以支持 HDR 色调映射
echo ============================================================
echo.
echo 当前版本:
for /f "tokens=*" %%a in ('"%~dp0ffmpeg\ffmpeg.exe" -version 2^>nul ^| findstr /i "ffmpeg"') do echo   %%a
echo.
echo 将下载约 100MB 并自动更新到 %~dp0ffmpeg\
echo 旧版将备份到 %~dp0ffmpeg_backup\
echo.
echo 请选择下载源：
echo   1. Gyan.dev 完整版（推荐）
echo   2. BtbN GitHub（备选）
echo   3. 手动模式（自行下载）
echo.
set /p CHOICE="请输入 (1/2/3): "

if "%CHOICE%"=="1" (
    set URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z
    set FNAME=ffmpeg-release-full-shared.7z
    goto :DOWNLOAD
)
if "%CHOICE%"=="2" (
    echo 暂不支持自动下载，请选择 3 手动模式
    pause
    exit /b 1
)
if "%CHOICE%"=="3" goto :MANUAL
echo 无效选择
pause
exit /b 1

:DOWNLOAD
echo.
echo 下载中… 请耐心等待（约 100MB）
echo 来源: %URL%
echo.
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%URL%' -OutFile '%temp%\%FNAME%' -Verbose}"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 下载失败！请手动下载：
    echo   %URL%
    echo 然后再次运行本脚本选择 3 手动模式
    pause
    exit /b 1
)
echo 下载完成！
set ARCHIVE=%temp%\%FNAME%
goto :EXTRACT

:MANUAL
echo.
echo 请手动下载完整版 FFmpeg：
echo   1. 打开 https://www.gyan.dev/ffmpeg/builds/
echo   2. 下载 ffmpeg-release-full-shared.7z
echo   3. 放到 %temp%\ 目录下
echo.
set /p ARCHIVE="输入 7z 文件路径（直接回车=%temp%\ffmpeg-release-full-shared.7z）: "
if "%ARCHIVE%"=="" set ARCHIVE=%temp%\ffmpeg-release-full-shared.7z
if not exist "%ARCHIVE%" (
    echo 文件不存在: %ARCHIVE%
    pause
    exit /b 1
)

:EXTRACT
echo.
echo 备份旧版到 %~dp0ffmpeg_backup\
if exist "%~dp0ffmpeg_backup" rmdir /s /q "%~dp0ffmpeg_backup"
if exist "%~dp0ffmpeg" xcopy "%~dp0ffmpeg" "%~dp0ffmpeg_backup\" /E /I /Q >nul

echo 解压中…
if not exist "%temp%\ffmpeg_extract" mkdir "%temp%\ffmpeg_extract"
"%~dp0ffmpeg\7z.exe" x "%ARCHIVE%" -o"%temp%\ffmpeg_extract" -y >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    REM 尝试系统的 7z
    7z x "%ARCHIVE%" -o"%temp%\ffmpeg_extract" -y >nul 2>&1
)
if %ERRORLEVEL% NEQ 0 (
    echo 解压失败！请安装 7-Zip 或手动解压后复制到 %~dp0ffmpeg\
    pause
    exit /b 1
)

REM 找到 ffmpeg.exe
set FFEXE=
for /r "%temp%\ffmpeg_extract" %%f in (*) do (
    if /i "%%~nxf"=="ffmpeg.exe" set FFEXE=%%~dpf
)
if "%FFEXE%"=="" (
    echo 未找到 ffmpeg.exe
    pause
    exit /b 1
)

echo 复制文件到 %~dp0ffmpeg\
REM 清空目标
if exist "%~dp0ffmpeg" rmdir /s /q "%~dp0ffmpeg"
mkdir "%~dp0ffmpeg"
xcopy "%FFEXE%" "%~dp0ffmpeg" /E /I /Q >nul

echo.
echo ============================================================
echo       ✅ 更新完成！
echo ============================================================
for /f "tokens=*" %%a in ('"%~dp0ffmpeg\ffmpeg.exe" -version 2^>nul ^| findstr /i "ffmpeg"') do echo   新版本: %%a

REM 检查 libplacebo
"%~dp0ffmpeg\ffmpeg.exe" -filters 2>nul | findstr "libplacebo" >nul
if %ERRORLEVEL% EQU 0 (
    echo   ✅ libplacebo 可用！支持 HDR 色调映射
) else (
    echo   ❌ libplacebo 不可用，请下载完整版
)

REM 清理
if exist "%temp%\ffmpeg_extract" rmdir /s /q "%temp%\ffmpeg_extract"
del "%ARCHIVE%" 2>nul

echo.
pause
