@echo off
chcp 65001 >nul
title Merger (ONNX DFM + SAXSeg ONNX)
cd /d C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main

mkdir "workspace\data_dst\merged" 2>nul
mkdir "workspace\data_dst\merged_mask" 2>nul

python -c "import subprocess,sys,glob; from pathlib import Path; root=Path(r'C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main'); dfm_list=glob.glob(str(root/'workspace'/'model'/'*.dfm')); dfm=dfm_list[0]; xseg=root/'workspace'/'model'/'SAXSeg'/'SAXSeg_256.onnx'; cmd=[sys.executable,'main.py','merge','--input-dir',str(root/'workspace'/'data_dst'),'--output-dir',str(root/'workspace'/'data_dst'/'merged'),'--output-mask-dir',str(root/'workspace'/'data_dst'/'merged_mask'),'--aligned-dir',str(root/'workspace'/'data_dst'/'aligned'),'--model-dir',str(root/'workspace'/'model'),'--model','SAEHD','--dfm-onnx',dfm,'--xseg-onnx',str(xseg),'--xseg-onnx-res','256']; print('DFM:',dfm); print('XSeg:',xseg); subprocess.run(cmd,cwd=str(root))"
pause
