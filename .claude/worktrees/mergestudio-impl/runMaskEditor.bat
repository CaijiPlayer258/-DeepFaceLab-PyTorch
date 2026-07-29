@echo off
REM Launch SAM Interactive Mask Editor

cd /d "%~dp0.."

echo Starting SAM Mask Editor...
python\Scripts\python.exe MaskProcessor\MaskEditorUI.py

pause
