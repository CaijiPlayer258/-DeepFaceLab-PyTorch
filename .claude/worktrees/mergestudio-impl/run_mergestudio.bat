@echo off
chcp 65001 >nul
title MergeStudio

echo ========================================
echo   MergeStudio - Starting...
echo ========================================
echo.

cd /d "%~dp0"

echo [*] Server: http://127.0.0.1:8000
echo [*] Press Ctrl+C to stop
echo.

python -m MergeStudio

if %errorlevel% neq 0 (
    echo.
    echo [!] Server stopped with error code %errorlevel%
    pause
)
