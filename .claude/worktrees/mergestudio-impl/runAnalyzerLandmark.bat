@echo off
chcp 65001 >nul
echo ===========================================================================
echo   Analyzer - Landmark 模式（恢复人脸特征点）
echo ===========================================================================
echo.

:menu
echo 请选择操作：
echo.
echo 1. 恢复特征点（仅 landmark）
echo 2. 恢复特征点 + 计算姿态（landmark + pose）
echo 3. 测试 landmark 功能
echo 4. 查看帮助
echo 5. 退出
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" goto restore_landmarks
if "%choice%"=="2" goto restore_landmarks_pose
if "%choice%"=="3" goto test_landmark
if "%choice%"=="4" goto show_help
if "%choice%"=="5" goto end

echo 无效选项，请重新选择
echo.
goto menu

:restore_landmarks
echo.
echo ===========================================================================
echo   恢复人脸特征点（仅 landmark）
echo ===========================================================================
echo.

set /p input_path="请输入人脸集目录路径: "

if "%input_path%"=="" (
    echo 错误：路径不能为空
    pause
    goto menu
)

if not exist "%input_path%" (
    echo 错误：目录不存在
    pause
    goto menu
)

echo.
echo 开始恢复特征点...
echo 目录: %input_path%
echo.

python "%~dp0Analyzer.py" -i "%input_path%" --features landmark

echo.
echo 完成！
pause
goto menu

:restore_landmarks_pose
echo.
echo ===========================================================================
echo   恢复人脸特征点 + 计算姿态
echo ===========================================================================
echo.

set /p input_path="请输入人脸集目录路径: "

if "%input_path%"=="" (
    echo 错误：路径不能为空
    pause
    goto menu
)

if not exist "%input_path%" (
    echo 错误：目录不存在
    pause
    goto menu
)

echo.
echo 开始恢复特征点并计算姿态...
echo 目录: %input_path%
echo.

python "%~dp0Analyzer.py" -i "%input_path%" --features landmark,pose

echo.
echo 完成！
pause
goto menu

:test_landmark
echo.
echo ===========================================================================
echo   测试 Landmark 功能
echo ===========================================================================
echo.

python "%~dp0test_landmark_mode.py"

echo.
pause
goto menu

:show_help
echo.
echo ===========================================================================
echo   帮助信息
echo ===========================================================================
echo.
echo 基本用法：
echo   python Analyzer.py -i ^<输入目录^> --features landmark
echo.
echo 可选参数：
echo   --force              强制重新分析（覆盖已有数据）
echo   --workers N          指定工作进程数（默认自动检测）
echo   --write-back         将元数据回写到 JPG 文件
echo.
echo 示例：
echo   python Analyzer.py -i ".\workspace\data_dst\aligned" --features landmark
echo   python Analyzer.py -i ".\workspace\data_dst\aligned" --features landmark,pose
echo   python Analyzer.py -i ".\workspace\data_dst\aligned" --features landmark --workers 8
echo.
echo 详细说明请查看：LANDMARK_MODE_USAGE.md
echo.
pause
goto menu

:end
echo.
echo 感谢使用！
exit /b 0
