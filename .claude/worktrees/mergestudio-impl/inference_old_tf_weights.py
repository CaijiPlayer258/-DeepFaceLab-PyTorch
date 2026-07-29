#!/usr/bin/env python3
"""
DFL 老版 TF .npy → PyTorch 通用推理
===================================
加载原版 DeepFaceLab (TF/Leras) 训练的 .npy 权重，在 PyTorch 中推理。

核心修复: TF depth_to_space (NCHW) vs PT pixel_shuffle 通道排列差异
- TF:  output[n,c,h·r+i,w·r+j] = input[n, (i·r+j)·C + c, h, w]
- PT:  output[n,c,h·r+i,w·r+j] = input[n, c·r² + i·r + j, h, w]
TF 先按空间子位置(i,j)分组,每组 C 个连续输出通道
PT 先按输出通道 c 分组,每个通道有 r² 个子位置
修复: perm[c·r² + i·r + j] = (i·r+j)·C + c, 再调 pixel_shuffle

用法:
    python inference_old_tf_weights.py
在下面 MODEL_PREFIX 处切换模型:
  "haixiu7"    — df-ud   320  ae256 e64  d64  dm22
  "jiulin"     — df-ud   320  ae256 e64  d64  dm22
  "DF-UDT416"  — df-udt  416  ae320 e80  d80  dm26
"""
import sys, pickle, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

REPO = Path(r"C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main")
WORKSPACE = REPO / "workspace" / "model"
sys.path.insert(0, str(REPO))

# ── 加载转换脚本，复用 shape 转换函数 ─────
spec = importlib.util.spec_from_file_location("converter",
    REPO / "tools" / "convert_dfl_tf_to_torch.py")
conv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = conv
spec.loader.exec_module(conv)
_convert_array_to_shape = conv._convert_array_to_shape

from core.leras import nn
nn.initialize(nn.DeviceConfig.CPU(), data_format="NCHW")

# ── 模型选择 ────────────────────────────────
MODEL_PREFIX = "jiulin"  # ← "haixiu7" / "jiulin" / "DF-UDT416"

data_path = WORKSPACE / f"{MODEL_PREFIX}_SAEHD_data.dat"
if not data_path.exists():
    raise FileNotFoundError(f"{data_path} 不存在")
options = pickle.loads(data_path.read_bytes()).get("options", {})
RES = int(options.get("resolution", 320))
AE_DIMS = int(options.get("ae_dims", 256))
E_DIMS = int(options.get("e_dims", 64))
D_DIMS = int(options.get("d_dims", 64))
DM_DIMS = int(options.get("d_mask_dims", 22))
ARCHI_STR = str(options.get("archi", "df-ud"))
ARCHI_OPTS = ARCHI_STR.split("-")[1] if "-" in ARCHI_STR else ""
print(f"Model: {MODEL_PREFIX}  archi={ARCHI_STR}  res={RES}")
print(f"  ae={AE_DIMS} e={E_DIMS} d={D_DIMS} dm={DM_DIMS}")

# ── TF 兼容 depth_to_space ─────────────────
def depth_to_space_tf(x, block_size):
    """TF-compatible depth_to_space (NCHW)."""
    N, C_in, H, W = x.shape
    r = block_size
    C_out = C_in // (r * r)
    idx = torch.arange(C_in, device=x.device)
    c = idx // (r * r)
    ij = idx % (r * r)
    i = ij // r
    j = ij % r
    perm = (i * r + j) * C_out + c  # TF→PT 通道置换
    return F.pixel_shuffle(x[:, perm, :, :], r)

nn.depth_to_space = depth_to_space_tf

# ── 构建模型 ────────────────────────────────
model_archi = nn.DeepFakeArchi(RES, opts=ARCHI_OPTS)
encoder = model_archi.Encoder(in_ch=3, e_ch=E_DIMS, name="encoder")
inter = model_archi.Inter(
    in_ch=encoder.get_out_ch() * encoder.get_out_res(RES) ** 2,
    ae_ch=AE_DIMS, ae_out_ch=AE_DIMS, name="inter")
decoder_src = model_archi.Decoder(
    in_ch=inter.get_out_ch(), d_ch=D_DIMS, d_mask_ch=DM_DIMS, name="decoder_src")
decoder_dst = model_archi.Decoder(
    in_ch=inter.get_out_ch(), d_ch=D_DIMS, d_mask_ch=DM_DIMS, name="decoder_dst")
for m in (encoder, inter, decoder_src, decoder_dst):
    m.build()

# ── 加载 .npy 权重 ─────────────────────────
def load_npy(path):
    d = np.load(str(path), allow_pickle=True)
    if isinstance(d, np.ndarray) and d.dtype == np.object:
        return d.item()
    return d

for scope, mod in [("encoder", encoder), ("inter", inter),
                    ("decoder_src", decoder_src), ("decoder_dst", decoder_dst)]:
    fn = f"{MODEL_PREFIX}_SAEHD_{scope}.npy"
    fp = WORKSPACE / fn
    if not fp.exists():
        print(f"  [SKIP] {scope}: {fn} 不存在")
        continue
    tf_dict = load_npy(fp)
    tf_names = mod._get_tf_weight_names()
    n_ok = 0
    for tf_name, tensor in tf_names:
        if tf_name in tf_dict:
            raw = np.array(tf_dict[tf_name])
            converted = _convert_array_to_shape(raw, tuple(tensor.shape))
            tensor.data.copy_(torch.from_numpy(converted.astype(np.float32)))
            n_ok += 1
    print(f"  {scope}: {n_ok}/{len(tf_names)}")

# ── 推理 ────────────────────────────────────
img = cv2.imread(str(WORKSPACE / "target.jpg"))
img = cv2.resize(img, (RES, RES))
img_f = img.astype(np.float32) / 255.0
inp_t = torch.from_numpy(img_f.transpose((2, 0, 1))[None, ...]).to(nn.device)

with torch.no_grad():
    code = inter(encoder(inp_t))
    pred_face, pred_mask = decoder_src(code)

out = pred_face[0].cpu().numpy().transpose(1, 2, 0)  # HWC BGR
mask = pred_mask[0, 0].cpu().numpy()

print(f"  raw output: mean={out.mean():.4f} std={out.std():.4f}")

# ── 空间偏移修正 ──────────────────────────
# 经交互测试确认的各模型偏移补偿值 (dy, dx)
OFFSET_MAP = {
    "haixiu7":    (17, 16),
    "jiulin":     (17, 16),
    "DF-UDT416":  (32, 29),
}
dy, dx = OFFSET_MAP.get(MODEL_PREFIX, (0, 0))
if dy or dx:
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(out, M_shift, (RES, RES),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M_shift, (RES, RES),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    print(f"  offset corrected: ({dy}, {dx})")

# ── 保存 ────────────────────────────────────
out_rgb = (np.clip(out[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
inp_rgb = (img_f[:, :, ::-1] * 255).astype(np.uint8)
mask3 = np.stack([mask] * 3, axis=-1)
comp = out * mask3 + img_f * (1 - mask3)
comp_rgb = (np.clip(comp[:, :, ::-1], 0, 1) * 255).astype(np.uint8)

Image.fromarray(out_rgb).save(str(WORKSPACE / "predicted_face.png"))
Image.fromarray((mask * 255).astype(np.uint8)).save(str(WORKSPACE / "predicted_mask.png"))
Image.fromarray(comp_rgb).save(str(WORKSPACE / "composited_result.png"))

# 四联图
gap = 8
canvas = np.zeros((RES + 28, RES * 4 + gap * 3, 3), dtype=np.uint8)
canvas[:RES, :RES] = inp_rgb
canvas[:RES, RES + gap : 2 * RES + gap] = out_rgb
canvas[:RES, 2 * RES + 2 * gap : 3 * RES + 2 * gap] = cv2.cvtColor(
    (mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
canvas[:RES, 3 * RES + 3 * gap :] = comp_rgb
for i, lbl in enumerate(["Input", "Output(SRC)", "Mask", "Composited"]):
    cv2.putText(canvas, lbl, (i * (RES + gap) + 4, RES + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
Image.fromarray(canvas).save(str(WORKSPACE / "inference_result.png"))

# 验证
gray = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY)
fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = fc.detectMultiScale(gray, 1.05, 3)
r_ch, g_ch = out[:, :, 0].ravel(), out[:, :, 1].ravel()
gx = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)).mean()
print(f"  faces={len(faces)}  RG corr={np.corrcoef(r_ch, g_ch)[0,1]:.4f}  grad={gx:.1f}")
print("Done!  Outputs in workspace/model/")
