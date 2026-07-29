# DeepFaceLab-Torch 安装指南

本文档详细说明如何安装 DeepFaceLab-Torch 项目所需的所有依赖库。

## 系统要求

- **Python**: 3.12 或更高版本（推荐 3.12.10）
- **操作系统**: Windows 10/11
- **GPU**: NVIDIA GPU（支持 CUDA 12.6+）
- **CUDA**: 12.6 或更高版本（需要更新 NVIDIA 驱动到最新版本）

## 安装步骤

### 1. 检查 Python 版本

```bash
python --version
```

确保 Python 版本 >= 3.12。如果不是，请从 https://www.python.org/downloads/ 下载安装。

### 2. （可选）创建虚拟环境

建议使用虚拟环境来隔离项目依赖：

```bash
# 在项目根目录执行
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
```

### 3. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 4. 安装 PyTorch（CUDA 版本）

**重要**: 必须先安装 PyTorch，因为它是一个大型包。

```bash
# 使用 CUDA 12.6 版本（推荐）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

如果你的 CUDA 版本不同，可以选择其他版本：
- CUDA 12.8: `--index-url https://download.pytorch.org/whl/cu128`
- CUDA 13.0: `--index-url https://download.pytorch.org/whl/cu130`

查看 PyTorch 官方安装指南: https://pytorch.org/get-started/locally/

### 5. 验证 PyTorch 安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

应该输出类似：
```
PyTorch version: 2.11.0+cu126
CUDA available: True
```

### 6. 安装其他依赖

```bash
# 安装 requirements.txt 中的所有依赖
python -m pip install -r requirements.txt
```

这将安装以下主要库：
- numpy, scipy, opencv-python（图像处理）
- onnxruntime, onnx（模型推理）
- facexlib（人脸提取）
- h5py（数据存储）
- scikit-learn（机器学习工具）
- timm, transformers（深度学习模型）
- albumentations（数据增强）
- PyQt5, pyperclip（GUI界面）
- 等等...

### 7. 安装 PyQt-SiliconUI

这是项目的 UI 框架，需要单独安装：

```bash
cd PyQt-SiliconUI-main
python -m pip install -e .
cd ..
```

### 8. （可选）安装 insightface

`insightface` 是一个强大的人脸识别库，但在 Windows 上需要从源码编译 C++ 扩展。

#### 方法 A: 使用 Conda（推荐）

如果你使用 Conda：

```bash
conda install -c conda-forge insightface
```

#### 方法 B: 安装 Visual C++ Build Tools

1. 下载 Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 安装时选择 "C++ build tools" 工作负载
3. 确保包含 "MSVC v142 - VS 2019 C++ x64/x86 build tools"
4. 重新运行: `python -m pip install insightface`

#### 方法 C: 跳过 insightface

如果不需要 insightface 功能，可以跳过此步骤。项目的大部分功能仍然可以正常使用。

### 9. 验证安装

运行以下命令测试基本功能：

```bash
# 检查主要依赖
python -c "import cv2; import torch; import onnxruntime; import PyQt5; print('All imports successful!')"

# 运行 Qt 自检
python main.py qt selftest
```

### 10. 启动程序

```bash
# 方法 1: 使用 run.bat（Windows）
.\run.bat

# 方法 2: 直接运行 Python
python .\ui\start.py

# 方法 3: 使用命令行界面
python main.py -h
```

## 常见问题

### Q1: 遇到 "DLL load failed" 错误

**解决方案**: 
- 确保已安装最新的 Visual C++ Redistributable
- 项目已自动处理 DLL 加载问题（见 `ui/start.py`）

### Q2: insightface 安装失败

**解决方案**:
- 使用 Conda 安装（方法 A）
- 或安装 Visual C++ Build Tools（方法 B）
- 或暂时跳过，使用 facexlib 作为替代

### Q3: CUDA 不可用

**解决方案**:
1. 确认 NVIDIA 驱动已更新到最新版本
2. 检查 CUDA 版本: `nvidia-smi`
3. 确保安装的 PyTorch CUDA 版本与系统 CUDA 兼容
4. 如果仍无法解决，可以使用 CPU 模式（速度较慢）:
   ```bash
   python main.py train --cpu-only ...
   ```

### Q4: 内存不足

**解决方案**:
- 减小 batch size
- 降低图像分辨率
- 关闭其他占用显存的程序

### Q5: PyQt5 导入错误

**解决方案**:
```bash
# 重新安装 PyQt5
python -m pip uninstall PyQt5
python -m pip install PyQt5==5.15.11
```

## 已安装的依赖列表

以下是项目使用的主要依赖库：

### 核心库
- torch (2.11.0+cu126)
- torchvision (0.26.0+cu126)
- numpy (2.4.3)
- opencv-python (4.13.0.92)
- scipy (1.17.1)
- Pillow (12.1.1)

### 深度学习
- onnxruntime (1.25.1)
- onnx (1.21.0)
- timm (1.0.26)
- transformers (5.7.0)
- facexlib (0.3.0)

### 数据处理
- h5py (3.16.0)
- scikit-learn (1.8.0)
- albumentations (2.0.8)
- numba (0.65.1)

### GUI
- PyQt5 (5.15.11)
- PyQt-SiliconUI (1.0.1)
- pyperclip (1.11.0)

### 工具库
- tqdm (4.67.3)
- requests (2.33.1)
- PyYAML (6.0.3)
- matplotlib (3.10.9)
- cython (3.2.4)

## 更新依赖

如果需要更新所有依赖到最新版本：

```bash
python -m pip install --upgrade -r requirements.txt
```

## 卸载

如果需要清理环境：

```bash
# 删除虚拟环境（如果使用）
rm -rf venv  # Linux/Mac
rmdir /s /q venv  # Windows

# 或手动卸载包
python -m pip uninstall -y -r requirements.txt
```

## 技术支持

如果遇到问题：
1. 检查 README.md 中的最新说明
2. 查看项目的 GitHub Issues
3. 确认 Python 和 CUDA 版本兼容性
4. 尝试在干净的虚拟环境中重新安装

## 参考链接

- PyTorch 官方: https://pytorch.org/
- DeepFaceLab 上游: https://github.com/iperov/DeepFaceLab
- PyQt-SiliconUI: https://github.com/ChinaIceF/PyQt-SiliconUI
