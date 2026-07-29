@echo off
chcp 65001 >nul
cd /d "%~dp0"


.\python\Scripts\python.exe Extractor\Extractor.py -i ".\workspace\data_dst" -o ".\workspace\data_dst\aligned" -d YoloV8Face -l insightface-2d106det -a 0,180

pause
