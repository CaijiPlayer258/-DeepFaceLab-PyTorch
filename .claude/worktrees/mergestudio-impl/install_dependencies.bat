@echo off
REM DeepFaceLab-Torch 快速安装脚本 (Windows)
REM 此脚本会自动安装所有必需的依赖库

echo ========================================
echo DeepFaceLab-Torch 依赖安装程序
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python。请先安装 Python 3.12 或更高版本。
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/6] 检查 Python 版本...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo 当前 Python 版本: %PYTHON_VERSION%
echo.

REM 升级 pip
echo [2/6] 升级 pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [错误] pip 升级失败
    pause
    exit /b 1
)
echo.

REM 安装 PyTorch
echo [3/6] 安装 PyTorch (CUDA 12.6 版本)...
echo 这可能需要几分钟时间，请耐心等待...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [错误] PyTorch 安装失败
    pause
    exit /b 1
)
echo.

REM 验证 PyTorch
echo [4/6] 验证 PyTorch 安装...
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
echo.

REM 安装其他依赖
echo [5/6] 安装其他依赖库...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [警告] 部分依赖安装可能失败，请检查上面的错误信息
)
echo.

REM 安装 PyQt-SiliconUI
echo [6/6] 安装 PyQt-SiliconUI...
cd PyQt-SiliconUI-main
python -m pip install -e .
if errorlevel 1 (
    echo [错误] PyQt-SiliconUI 安装失败
    cd ..
    pause
    exit /b 1
)
cd ..
echo.

REM 最终验证
echo ========================================
echo 安装完成！正在验证...
echo ========================================
python -c "import cv2, torch, onnxruntime, PyQt5; print('✓ 所有核心依赖导入成功！')"
echo.

echo ========================================
echo 安装成功！
echo ========================================
echo.
echo 下一步：
echo 1. 运行 GUI: run.bat
echo 2. 或使用命令行: python main.py -h
echo.
echo 如需安装 insightface（可选）：
echo   - 使用 Conda: conda install -c conda-forge insightface
echo   - 或安装 Visual C++ Build Tools 后运行: pip install insightface
echo.
pause
