@echo off
chcp 65001 >nul
title Mask Processor

echo ========================================
echo   Mask Processor - Starting...
echo ========================================
echo.

cd /d "%~dp0"

echo [*] Activating Conda environment...
call C:\Users\nobody\anaconda3\Scripts\activate.bat >nul 2>&1

echo [*] Starting server at http://127.0.0.1:8000
echo [*] Press Ctrl+C to stop
echo.

python -m MaskProcessor

if %errorlevel% neq 0 (
    echo.
    echo [!] Server stopped with error code %errorlevel%
    echo [!] Check the output above for details
    pause
)
