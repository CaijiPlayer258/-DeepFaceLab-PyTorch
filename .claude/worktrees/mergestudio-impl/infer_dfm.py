"""
DFM (ONNX) 推理演示脚本

用法:
    python infer_dfm.py <模型.dfm> <输入图片>

示例:
    python infer_dfm.py workspace/model/幼蓝蓝_SAEHD_model.dfm workspace/model/target.jpg
"""

import os
import sys

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


def main():
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
    else:
        model_path = input("模型路径: ").strip('"').strip("'").strip()
        image_path = input("图片路径: ").strip('"').strip("'").strip()

    # --- 加载 ONNX 模型（BF16 模型需 CUDA EP） ---
    print(f"加载模型: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception:
        session = ort.InferenceSession(model_path)

    inp = session.get_inputs()[0]
    input_name = inp.name
    input_shape = inp.shape  # (1, H, W, 3) NHWC
    target_size = input_shape[1]

    n_inputs = len(session.get_inputs())
    output_names = [o.name for o in session.get_outputs()]
    print(f"  输入: {input_name}, shape: {input_shape}")
    print(f"  输出: {output_names}")

    # --- 加载图片 ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片加载失败: {image_path}")
        sys.exit(1)

    img_resized = cv2.resize(img, (target_size, target_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = img_normalized[np.newaxis, ...]  # (1, H, W, 3)

    # --- 推理 ---
    print("推理中 ...")
    if n_inputs == 2:
        morph_name = session.get_inputs()[1].name
        morph_val = np.array([1.0], dtype=np.float32)
        outputs = session.run(None, {input_name: img_input, morph_name: morph_val})
    else:
        outputs = session.run(None, {input_name: img_input})

    # --- 显示结果 ---
    # 约定输出顺序: [out_face_mask, out_celeb_face, out_celeb_face_mask]
    out_face_mask = outputs[0]       # (1, H, W, 1) NHWC
    out_celeb_face = outputs[1]      # (1, H, W, 3) NHWC
    out_celeb_face_mask = outputs[2] if len(outputs) > 2 else out_face_mask

    disp_face = (out_celeb_face[0] * 255).clip(0, 255).astype(np.uint8)
    disp_mask = (out_face_mask[0] * 255).clip(0, 255).astype(np.uint8)
    disp_mask2 = (out_celeb_face_mask[0] * 255).clip(0, 255).astype(np.uint8)

    # 组合显示
    mask_3ch = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)
    mask2_3ch = cv2.cvtColor(disp_mask2, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack((img_resized, disp_face))
    bot_row = np.hstack((mask_3ch, mask2_3ch))
    canvas = np.vstack((top_row, bot_row))

    cv2.imshow("DFM Inference (input | celeb_face  |  mask | celeb_mask)", canvas)
    print("按任意键关闭窗口 ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
