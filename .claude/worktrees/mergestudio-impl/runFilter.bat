@echo off
chcp 65001 >nul
cd /d "%~dp0"


.\python\Scripts\python.exe FacesetProcessor\Filter.py -i ".\workspace\data_dst\aligned" --mode faceid


pause
