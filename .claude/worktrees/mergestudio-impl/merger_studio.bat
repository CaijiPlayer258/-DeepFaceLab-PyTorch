@echo off
chcp 65001 >nul
title Merger Studio
cd /d C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main
echo Merger Studio starting...
echo http://localhost:6789/MergerStudio
python merger_studio.py
pause
