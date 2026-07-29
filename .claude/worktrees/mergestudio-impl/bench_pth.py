"""
PyTorch 原生 BF16 推理速度测试 + 可视化

用法:
    python bench_pth.py <模型名> [图片目录]

示例:
    python bench_pth.py 幼蓝蓝_SAEHD workspace/data_dst/aligned
"""

import os
import sys
import time
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
os.chdir(str(_BASE_DIR))
sys.path.insert(0, str(_BASE_DIR))

import cv2
import numpy as np
import torch
import models
from core.leras import nn


nn.initialize_main_env()


def draw_fps_overlay(img, fps, ms):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (260, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, f"FPS: {fps:.1f}  ({ms:.1f} ms)", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <模型名> [图片目录]")
        print(f"示例: python {sys.argv[0]} 幼蓝蓝_SAEHD workspace/data_dst/aligned")
        sys.exit(1)

    model_name = sys.argv[1]
    img_dir = sys.argv[2] if len(sys.argv) >= 3 else "workspace/data_dst/aligned"

    # --- 加载 PyTorch 模型（与训练同样的加载方式）---
    print(f"加载模型: {model_name}")
    model_cls = models.import_model("SAEHD")
    model = model_cls(
        is_exporting=False,
        saved_models_path=Path("workspace/model").resolve(),
        cpu_only=False,
        silent_start=True,
        force_model_name=model_name,
    )

    resolution = model.resolution
    archi = model.archi_type
    print(f"  分辨率: {resolution}, 架构: {archi}")
    print(f"  训练迭代: {model.get_iter()}")
    print(f"  总参数: {sum(p.numel() for p in model.encoder.parameters() if hasattr(p, 'numel')) * 4:,}")

    # --- 切换到 eval 模式 ---
    model.encoder.eval()
    model.inter.eval()
    model.decoder_src.eval()
    model.decoder_dst.eval()

    # --- 将所有权重转为 BF16 并移到 GPU ---
    device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    print(f"  推理设备: {device}")

    dtype = torch.bfloat16
    model.encoder.to(device, dtype=dtype)
    model.inter.to(device, dtype=dtype)
    model.decoder_src.to(device, dtype=dtype)
    model.decoder_dst.to(device, dtype=dtype)
    print(f"  推理精度: BF16")

    # --- 扫描图片 ---
    img_paths = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
    if not img_paths:
        print(f"  目录中无图片: {img_dir}")
        sys.exit(1)
    print(f"  找到 {len(img_paths)} 张图片")

    # --- Warmup ---
    dummy = torch.zeros((1, 3, resolution, resolution), dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(5):
            code = model.inter(model.encoder(dummy))
            _ = model.decoder_src(code)

    # --- Benchmark + 可视化 ---
    times = []
    count = 0
    window_name = "PyTorch BF16 Benchmark - 按 ESC 退出"

    print("\n开始测试（按 ESC 或 Ctrl+C 提前结束）...")
    sys.stdout.flush()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fps_display = 0.0
    ms_display = 0.0

    try:
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_resized = cv2.resize(img, (resolution, resolution))

            # NHWC uint8 → NCHW BF16 normalized
            img_input = torch.from_numpy(img_resized).to(device, dtype=dtype) / 255.0
            img_input = img_input.permute(2, 0, 1).unsqueeze(0).contiguous()

            t0 = time.perf_counter()
            with torch.no_grad():
                code = model.inter(model.encoder(img_input))
                out_celeb_face, out_celeb_face_mask = model.decoder_src(code)
                _, out_face_mask = model.decoder_dst(code)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            times.append(elapsed)
            count += 1

            recent = times[-min(count, 30):]
            ms_display = sum(recent) / len(recent) * 1000
            fps_display = 1000.0 / ms_display if ms_display > 0 else 0

            # --- 可视化（转回 CPU numpy）---
            def to_nhwc(t):
                return t.permute(0, 2, 3, 1).cpu().float().numpy()[0]

            out_celeb_face_np = to_nhwc(out_celeb_face).clip(0, 1)
            out_face_mask_np = to_nhwc(out_face_mask).clip(0, 1)
            celeb_mask_np = to_nhwc(out_celeb_face_mask).clip(0, 1)

            disp_face = (out_celeb_face_np * 255).astype(np.uint8)

            def to_3ch(m):
                m8 = (m * 255).astype(np.uint8)
                if m8.ndim == 2 or m8.shape[2] == 1:
                    return cv2.cvtColor(m8, cv2.COLOR_GRAY2BGR)
                return m8

            dst_mask_3ch = to_3ch(out_face_mask_np)
            celeb_mask_3ch = to_3ch(celeb_mask_np)

            mf = celeb_mask_np.astype(np.float32)
            if mf.ndim == 3 and mf.shape[2] == 1:
                mf = mf[..., 0]
            blended = (disp_face.astype(np.float32) * mf[..., None] +
                       img_resized.astype(np.float32) * (1.0 - mf[..., None])).clip(0, 255).astype(np.uint8)

            # 布局：5 列两行，不区分 src/dst 顺序差异
            cell_w = resolution
            cell_h = resolution
            n_cols = 5

            labels = ["Input", "Pred Face", "Blended", "DST Mask", "PRED Mask"]
            label_bar = np.ones((28, cell_w * n_cols, 3), dtype=np.uint8) * 48
            for ci, lbl in enumerate(labels):
                cv2.putText(label_bar, lbl, (ci * cell_w + 8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            row0 = np.hstack((img_resized, disp_face, blended, dst_mask_3ch, celeb_mask_3ch))
            row1 = np.hstack((img_resized, disp_face, blended, celeb_mask_3ch, dst_mask_3ch))

            canvas = np.vstack((label_bar, row0, row1))
            draw_fps_overlay(canvas, fps_display, ms_display)

            cv2.resizeWindow(window_name, cell_w * n_cols, (cell_h * 2) + 28)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\n  用户退出")
                break

            if count % 500 == 0:
                avg_ms = sum(times[-500:]) / len(times[-500:]) * 1000
                print(f"  {count}/{len(img_paths)}  ... {avg_ms:.1f} ms/it")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n  测试中断")

    cv2.destroyAllWindows()

    if times:
        avg_ms = sum(times) / len(times) * 1000
        fps = 1000 / avg_ms
        print(f"\n{'='*40}")
        print(f"  总张数:     {count}")
        print(f"  平均耗时:   {avg_ms:.2f} ms/it")
        print(f"  帧率:       {fps:.1f} FPS")
        if len(times) > 1:
            import statistics
            print(f"  标准差:     {statistics.stdev(times) * 1000:.2f} ms")
            print(f"  最小值:     {min(times) * 1000:.2f} ms")
            print(f"  最大值:     {max(times) * 1000:.2f} ms")
        print(f"  推理精度:   BF16")
        print(f"  推理设备:   {device}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
