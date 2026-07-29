"""
独立 DFM(ONNX) 导出脚本

用法:
    python export_dfm.py

支持 SAEHD / AMP 模型导出，可选择:
  - fp32   (标准精度)
  - bf16   (BrainFloat16 半精度，ONNX 内核算 BF16，图出入为 FP32)
  - 量化: 导出后再对 ONNX 模型进行动态量化 (int8)，进一步减小体积
"""

import os
import sys
import warnings
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
os.chdir(str(_BASE_DIR))
sys.path.insert(0, str(_BASE_DIR))

import models
from core.interact import interact as io


def get_input(prompt_text, default=None, valid_options=None):
    """简易控制台输入，支持默认值和选项校验。"""
    if valid_options:
        hint = f" ({'/'.join(valid_options)})"
    else:
        hint = ""
    default_hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt_text}{hint}{default_hint}: ").strip()
        if not raw and default is not None:
            return default
        if valid_options and raw not in valid_options:
            print(f"  请输入: {'/'.join(valid_options)}")
            continue
        return raw


def main():
    print("=" * 50)
    print("  DFM (ONNX) 导出工具")
    print("=" * 50)

    # --- 收集参数 ---
    model_dir = get_input("模型目录", default="workspace/model")
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        print(f"\n[错误] 目录不存在: {model_dir}")
        sys.exit(1)

    model_files = sorted(model_dir_path.glob("*_data.dat"))
    if not model_files:
        print(f"\n[错误] 目录中未找到 *_data.dat，请确认模型路径正确。")
        sys.exit(1)

    model_names = []
    for f in model_files:
        name = f.stem
        if name.endswith("_data"):
            name = name[:-5]
        model_names.append(name)

    print(f"\n可用模型:")
    for i, name in enumerate(model_names):
        print(f"  [{i}] {name}")
    default_idx = 0
    idx_input = input(f"\n选择模型索引 [0]: ").strip()
    if idx_input:
        try:
            selected_idx = int(idx_input)
            if selected_idx < 0 or selected_idx >= len(model_names):
                print(f"  索引超出范围，使用默认 [0]")
                selected_idx = 0
        except ValueError:
            print(f"  输入无效，使用默认 [0]")
            selected_idx = 0
    else:
        selected_idx = default_idx

    selected_model_name = model_names[selected_idx]
    print(f"  已选择: {selected_model_name}")

    model_type = get_input("模型类型", default="SAEHD", valid_options=["SAEHD", "AMP"])
    precision = get_input("导出精度", default="fp32", valid_options=["fp32", "fp16"])
    quantize = get_input("量化导出 (int8)", default="n", valid_options=["y", "n"])

    do_quantize = quantize.lower() == "y"
    use_fp16 = precision.lower() == "fp16"

    # 设置精度环境变量，模型 export_dfm() 会读取
    os.environ["DFM_EXPORT_PRECISION"] = "fp16" if use_fp16 else "fp32"
    os.environ["DFL_SILENT_INPUT"] = "1"

    print(f"\n正在加载模型 ({selected_model_name}) ...")
    sys.stdout.flush()

    warnings.filterwarnings(
        "ignore",
        message=r"Constant folding - Only steps=1 can be constant folded.*onnx::Slice.*",
        category=UserWarning,
    )

    try:
        model_cls = models.import_model(model_type)
        model = model_cls(
            is_exporting=True,
            saved_models_path=model_dir_path.resolve(),
            cpu_only=True,
            silent_start=True,
            force_model_name=selected_model_name,
        )
    except Exception as e:
        print(f"\n[错误] 模型加载失败: {e}")
        sys.exit(1)

    # --- 导出 ONNX ---
    prec_label = "FP16" if use_fp16 else "FP32"
    print(f"\n正在导出 ONNX ({prec_label}) ...")
    sys.stdout.flush()

    try:
        model.export_dfm()
    except Exception as e:
        print(f"\n[错误] ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 获取导出路径
    onnx_name = f"{selected_model_name}_model.dfm"
    onnx_path = model_dir_path / onnx_name
    if not onnx_path.exists():
        onnx_path = model_dir_path / f"{selected_model_name}_model.onnx"

    if not onnx_path.exists():
        print("\n  [警告] 未找到导出的 ONNX 文件，可能导出路径异常。")
        return

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    prec_label = "FP16" if use_fp16 else "FP32"
    print(f"  {prec_label} 导出完成: {onnx_path} ({size_mb:.1f} MB)")

    # --- 量化（后处理） ---
    if do_quantize:
        print("\n正在量化模型 (int8 dynamic) ...")
        sys.stdout.flush()
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            import shutil, tempfile
            # 中文文件名会导致 onnxruntime 内部 -inferred 中间文件编码错误
            tmp_dir = onnx_path.resolve().parent
            tmp_src = tmp_dir / "__quant_input.onnx"
            tmp_dst = tmp_dir / "__quant_output.onnx"
            shutil.copy2(str(onnx_path.resolve()), str(tmp_src))
            quantize_dynamic(
                str(tmp_src),
                str(tmp_dst),
                weight_type=QuantType.QUInt8,
            )
            qsize_mb = tmp_dst.stat().st_size / (1024 * 1024)
            print(f"  量化完成 ({qsize_mb:.1f} MB)")
            print(f"  压缩率: {qsize_mb / size_mb:.1%}")
            shutil.copy2(str(tmp_dst), str(onnx_path.resolve()))
            tmp_src.unlink(missing_ok=True)
            tmp_dst.unlink(missing_ok=True)
            print(f"  量化模型已覆盖: {onnx_path}")
        except ImportError:
            print("  [警告] onnxruntime 未安装，跳过量化。")
        except Exception as e:
            print(f"  [警告] 量化失败: {e}")

    print("\n" + "=" * 50)
    print("  导出完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
