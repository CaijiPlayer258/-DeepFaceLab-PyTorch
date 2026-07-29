@echo off
REM DeepFaceLab Torch - Faceset Sorter Launcher
REM 人脸集排序器启动脚本

cd /d "%~dp0.."

REM Check if input path is provided
if "%1"=="" (
    echo Usage: runSorter.bat ^<input_path^> [options]
    echo.
    echo Options:
    echo   --method, -m        Sort method (phash, hist, blur, face_pose, resolution, color, name)
    echo   --rename            Rename sorted files
    echo   --prefix            Rename prefix (default: sorted)
    echo   --workers           Number of worker processes
    echo   --pose-type         Pose type for face_pose (pitch, yaw, roll)
    echo   --motion-blur       Use motion blur detection
    echo.
    echo Examples:
    echo   runSorter.bat workspace\data_dst\aligned
    echo   runSorter.bat workspace\data_dst\aligned --method blur
    echo   runSorter.bat workspace\data_dst\aligned --method phash --rename
    echo   runSorter.bat workspace\data_dst\aligned --method face_pose --pose-type yaw
    exit /b 1
)

REM Run sorter with python
python FacesetProcessor\Sorter.py %*

pause
