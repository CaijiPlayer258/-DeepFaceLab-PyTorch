@echo off
chcp 65001 >nul
cd /d "%~dp0"


.\python\Scripts\python.exe FacesetProcessor\Analyzer.py -i ".\workspace\data_dst\aligned"


pause
