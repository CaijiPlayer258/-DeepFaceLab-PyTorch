"""
DFM (ONNX) 推理速度测试 + 可视化脚本

用法:
    python bench_dfm.py <模型.dfm> [图片目录]

示例:
    python bench_dfm.py workspace/model/幼蓝蓝_SAEHD_model.dfm workspace/data_dst/aligned
"""

import os
import sys
import time
from pathlib import Path

# ---- 自动查找 CUDA + cuDNN DLL 路径（onnxruntime-gpu 需要） ----
for _dir in [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    os.path.join(os.environ.get("USERPROFILE", ""), "anaconda3", "Library", "bin"),
    os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin"),
]:
    if os.path.isdir(_dir) and any(f.endswith(".dll") for f in os.listdir(_dir)):
        os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_dir)
        except Exception:
            pass

import cv2
import numpy as np
import onnxruntime as ort


def draw_fps_overlay(img, fps, ms):
    """在图片左上角叠加 FPS 信息"""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (260, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, f"FPS: {fps:.1f}  ({ms:.1f} ms)", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <模型.dfm> [图片目录]")
        sys.exit(1)

    model_path = sys.argv[1]
    img_dir = sys.argv[2] if len(sys.argv) >= 3 else "workspace/data_dst/aligned"

    # --- 加载 ONNX ---
    print(f"加载模型: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception:
        session = ort.InferenceSession(model_path)

    inp = session.get_inputs()[0]
    input_name = inp.name
    target_size = inp.shape[1]
    n_inputs = len(session.get_inputs())
    n_outputs = len(session.get_outputs())
    provider = session.get_providers()[0]
    print(f"  输入: {input_name}, {target_size}x{target_size}")
    print(f"  推理设备: {provider}")

    # --- 打印模型精度信息 ---
    inp_type = inp.type
    out_types = [o.type for o in session.get_outputs()]
    print(f"  输入类型: {inp_type}")
    print(f"  输出类型: {out_types}")

    # 通过检查 ONNX initializer 的 dtype 判断内部权重精度
    try:
        import onnx
        _m = onnx.load(str(model_path))
        _dtypes = {}
        for init in _m.graph.initializer:
            _dtypes[init.data_type] = _dtypes.get(init.data_type, 0) + 1
        _dtype_names = {1: "FP32", 10: "FP16", 16: "BF16"}
        _parts = []
        for dt, cnt in sorted(_dtypes.items()):
            name = _dtype_names.get(dt, f"unknown({dt})")
            _parts.append(f"{name}={cnt}")
        print(f"  权重精度: {', '.join(_parts)}")
    except Exception:
        pass

    _size_mb = os.path.getsize(str(model_path)) / (1024 * 1024)
    print(f"  文件大小: {_size_mb:.1f} MB")

    # --- 扫描图片 ---
    img_paths = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
    if not img_paths:
        print(f"  目录中无图片: {img_dir}")
        sys.exit(1)
    print(f"  找到 {len(img_paths)} 张图片")

    # --- Warmup ---
    dummy = np.zeros((1, target_size, target_size, 3), dtype=np.float32)
    for _ in range(5):
        if n_inputs == 2:
            session.run(None, {input_name: dummy, session.get_inputs()[1].name: np.array([1.0], dtype=np.float32)})
        else:
            session.run(None, {input_name: dummy})

    # --- Benchmark + 可视化 ---
    times = []
    count = 0
    window_name = "DFM Benchmark - 按 ESC 退出"

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
            img_resized = cv2.resize(img, (target_size, target_size))
            img_input = (img_resized.astype(np.float32) / 255.0)[np.newaxis, ...]

            t0 = time.perf_counter()
            if n_inputs == 2:
                outputs = session.run(None, {
                    input_name: img_input,
                    session.get_inputs()[1].name: np.array([1.0], dtype=np.float32)
                })
            else:
                outputs = session.run(None, {input_name: img_input})
            t1 = time.perf_counter()

            elapsed = t1 - t0
            times.append(elapsed)
            count += 1

            # 平滑 FPS 显示
            recent = times[-min(count, 30):]
            ms_display = sum(recent) / len(recent) * 1000
            fps_display = 1000.0 / ms_display if ms_display > 0 else 0

            # --- 可视化 ---
            out_celeb_face = outputs[1][0] if n_outputs >= 2 else outputs[0][0]
            dst_mask = outputs[0][0]       # out_face_mask
            celeb_mask = outputs[2][0] if n_outputs >= 3 else outputs[0][0]  # out_celeb_face_mask

            disp_face = (out_celeb_face * 255).clip(0, 255).astype(np.uint8)

            # 两个 mask 转 3 通道显示
            def _to_3ch(m):
                m8 = (m * 255).clip(0, 255).astype(np.uint8)
                if m8.ndim == 2 or m8.shape[2] == 1:
                    return cv2.cvtColor(m8, cv2.COLOR_GRAY2BGR)
                return m8

            dst_mask_3ch = _to_3ch(dst_mask)
            celeb_mask_3ch = _to_3ch(celeb_mask)

            # 换脸结果与原图融合（使用 celeb_mask）
            mf = celeb_mask.astype(np.float32)
            if mf.ndim == 3 and mf.shape[2] == 1:
                mf = mf[..., 0]
            blended = (disp_face.astype(np.float32) * mf[..., None] +
                       img_resized.astype(np.float32) * (1.0 - mf[..., None])).clip(0, 255).astype(np.uint8)

            # 布局：5 列两行
            cell_w = target_size
            cell_h = target_size
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
            if key == 27:  # ESC
                print("\n  用户退出")
                break

            if count % 500 == 0:
                avg_ms = sum(times[-500:]) / len(times[-500:]) * 1000
                print(f"  {count}/{len(img_paths)}  ... {avg_ms:.1f} ms/it")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n  测试中断")

    cv2.destroyAllWindows()

    # --- 结果 ---
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
        print(f"  推理设备:   {provider}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
