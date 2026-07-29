# deepfacelab_Torch

本仓库是一个 DeepFaceLab 相关的 PyTorch 版本实验/工程化仓库，重点包含：
- 训练 / 提取 / 合成等 CLI 入口（推荐使用 `main.py`）
- 原版 DeepFaceLab（TF/Leras）模型到本仓库 Torch Saveable 格式的无损转换工具（权重 + optimizer state）
- 人脸分割模型 XSegLite（Conv-BN-ReLU，3.0ms @ RTX 4060）
- 开源协作：请尊重上游与社区，反对“闭源加密后二次售卖”

> 重要：本仓库仅用于研究与学习目的。请在合法合规、尊重他人权利的前提下使用。

## 上游项目（必须注明）

DeepFaceLab 上游项目地址：
- https://github.com/iperov/DeepFaceLab

若你分发本仓库或其修改版，请在显著位置保留上游项目的来源说明。

## 预览

下列截图位于 `doc/` 目录：
- ![预览 1](doc/2026-01-01%20103041.png)
- ![预览 2](doc/2026-01-02%2015.35.40.png)
- ![预览 3](doc/2026-01-01%20103300.png)
- ![预览 4](doc/2026-01-02%2015.28.19.png)


## 验证环境

以下为作者环境的验证信息（用于复现参考）：
- Python：3.14 or 3.12
- PyTorch：2.9.x（见 `requirements.txt`）
- CUDA：13（只需要更新nvidia驱动到最新版本，确认支持cuda13就行）
- 已经确认cuda12.6 12.8 13都正常运行，根据官方https://pytorch.org/get-started/locally/安装PyTorch2.9.1 2.9.0即可

> 注：CUDA/驱动组合可能随平台变化。若你在不同环境上遇到安装/运行差异，建议优先以 PyTorch 官方提供的安装为准。

## 安装

### 1) 获取代码

- `git clone` 或直接下载 ZIP 均可。

### 2) 安装 Python 3.14.0

- 下载地址： https://www.python.org/downloads/

### 3) （可选）创建 venv

建议用 venv 做依赖隔离。

```bash
# 进入项目根目录
python3 -m venv venv
```

激活 venv：

```bash
# Linux / macOS
source ./venv/bin/activate
```

```bat
:: Windows（PowerShell / CMD 路径可能略有差异）
venv\Scripts\activate
```

升级 pip：

```bash
python3 -m pip install --upgrade pip
```

### 4) 更新nvidia到最新版本，安装 PyTorch（CUDA 13）

安装指引： https://pytorch.org/get-started/locally/

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

然后安装其余依赖：

```bash
# 建议先从 requirements.txt 中移除/注释 torch、torchvision，再执行：
python3 -m pip install -r requirements.txt
```

### 5) 基础自检

如果只想快速确认入口与依赖没问题：

```bash
python3 main.py -h
```

也可以运行交互启动器（如果你更喜欢菜单式操作）：

```bash
python3 run.py
```

## 推荐入口：用 main.py（对齐原版 bat 思路）

`run.py` 是交互式启动器，但你已经观察到其使用逻辑可能存在问题。
为了可脚本化、可复现、参数可控，建议直接使用：

```bash
python3 main.py -h
```

常用命令示例（对齐原版 bat 的“明确参数”用法）：

- 提取人脸（extract）：

```bash
python3 main.py extract \
  --input-dir  /ABS/PATH/to/frames_or_images \
  --output-dir /ABS/PATH/to/faceset
```

- 训练（train）：

```bash
python3 main.py train \
  --training-data-src-dir /ABS/PATH/to/workspace/data_src/aligned \
  --training-data-dst-dir /ABS/PATH/to/workspace/data_dst/aligned \
  --model-dir             /ABS/PATH/to/workspace/model \
  --model                 Model_SAEHD        # 或 Model_XSegLite
```

- 合成（merge）：

```bash
python3 main.py merge \
  --input-dir       /ABS/PATH/to/frames \
  --output-dir      /ABS/PATH/to/output \
  --output-mask-dir /ABS/PATH/to/output_mask \
  --model-dir       /ABS/PATH/to/workspace/model \
  --model           Model_SAEHD
```

你也可以使用 `--silent-start` 进行“静默启动”（自动选 GPU 与最近模型），详见 `python3 main.py train -h`。

## TF/Leras → Torch 模型转换（已验证可继续训练）

本仓库提供两份工具：
- 转换脚本：`tools/convert_dfl_tf_to_torch.py`
- 校验脚本：`tools/verify_converted_dfl_model.py`

### 1) 转换

```bash
python3 tools/convert_dfl_tf_to_torch.py \
  --src /ABS/PATH/to/original_dfl_saved_models \
  --dst /ABS/PATH/to/torch_saved_models \
  --all
```

转换内容包括：
- 网络权重（encoder/inter/decoder 等）
- 训练 optimizer state（用于继续训练稳定衔接）
- `*_data.dat` 与 `*_default_options.dat`

### 2) 校验（强烈建议）

```bash
python3 tools/verify_converted_dfl_model.py \
  --src /ABS/PATH/to/original_dfl_saved_models \
  --dst /ABS/PATH/to/torch_saved_models \
  --all
```

校验会做：
- 源权重字典对 Torch skeleton 的覆盖率检查（是否“全量覆盖”）
- 转换后 `.pth` 的 `param_0..param_n` 完整性检查
- 对每个模块/optimizer 执行一次真实 `load_weights()`（只加载，不训练）

## 许可与分发声明（务必阅读）

- 若你基于 DeepFaceLab 或其衍生代码进行分发/修改/再发布，请遵守上游项目的许可证要求，并保留/提供相应的许可证文件。
- 本仓库在文档层面明确**不鼓励商业使用**，并建议分发修改版时在显著位置注明来源与改动。

> 免责声明：以上为工程层面的说明与建议，不构成法律意见。许可证冲突/合规问题以许可证原文及适用法律为准。

## AI 辅助开发说明

本项目的大部分代码在开发过程中使用了 AI 工具辅助完成（用于加速代码梳理、脚本迭代与文档整理）。

## 本项目需要更多人参与完成

- 本人下班后利用闲余时间启动这个项目，精力有限；同时我对深度学习理解还比较浅。
- 如果你愿意一起完善，非常感谢。
