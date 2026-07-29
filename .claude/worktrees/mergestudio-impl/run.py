#!/usr/bin/env python3
"""DeepFaceLab Torch 版本启动器（原版 bat 工作流移植）。

本文件的目标是“移植原版 bat 的常用命令编排到 Python
因此：
- 交互菜单直接调用本仓库的 `main.py` 子命令
- 自动使用 workspace 目录的默认布局（data_src / data_dst / model）
- 支持 `--list` 列出菜单项、`--run` 直接运行某个菜单项
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ==============================================================================
# 可配置参数
# ==============================================================================
# 将 workspace 目录路径写死在这里（优先级仅次于命令行 --workspace）。
# 例：WORKSPACE_DIR = '/ABS/PATH/to/workspace'
WORKSPACE_DIR: str | None = None


@dataclass(frozen=True)
class LaunchContext:
    dfl_root: Path
    workspace: Path
    python_executable: str

@dataclass(frozen=True)
class MenuItem:
    key: str
    title: str
    group: str
    run: Callable[[LaunchContext], int]


def _open_path(path: Path) -> None:
    # 尝试用系统默认程序打开文件或目录（跨平台）
    # 感谢 cnddaofeng 提供错误反馈后补充
    # macOS: 使用 `open`
    if sys.platform == 'darwin':
        subprocess.run(['open', str(path)], check=False)
        return
    # Windows: 使用 os.startfile（仅 Windows 可用）
    if sys.platform == 'win32':
        os.startfile(str(path))
        return
    # 其他平台（Linux 等）：尝试使用 xdg-open 或 gio
    opener = shutil.which('xdg-open') or shutil.which('gio')
    if opener:
        if opener.endswith('gio'):
            subprocess.run([opener, 'open', str(path)], check=False)
        else:
            subprocess.run([opener, str(path)], check=False)


def _run_python_main(args: list[str], ctx: LaunchContext) -> int:
    main_py = ctx.dfl_root / 'main.py'
    if not main_py.exists():
        raise FileNotFoundError(f'未找到 main.py: {main_py}')

    cmd = [ctx.python_executable, str(main_py), *args]
    print('\n$ ' + ' '.join(_quote_for_display(x) for x in cmd))
    try:
        p = subprocess.run(cmd)
        return int(p.returncode)
    except KeyboardInterrupt:
        return 130


def _quote_for_display(s: str) -> str:
    if re.search(r'\s', s):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rm_tree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f'（默认：{default}）' if default else ''
    s = input(f'{text}{suffix}：').strip()
    return s if s else (default or '')


def _prompt_yes_no(text: str, default: bool = False) -> bool:
    hint = 'Y/n' if default else 'y/N'
    while True:
        s = input(f'{text} [{hint}]：').strip().lower()
        if not s:
            return default
        if s in ('y', 'yes'):
            return True
        if s in ('n', 'no'):
            return False


def _prompt_path(text: str, default: Path, ensure: bool = False) -> Path:
    s = _prompt(text, str(default))
    p = Path(os.path.expanduser(s)).resolve()
    if ensure:
        _ensure_dir(p)
    return p


def _find_default_pretraining_dir(workspace: Path) -> Path | None:
    """Try to find a sensible default pretraining faceset directory.

    Notes:
    - For DFL training CLI, we usually pass an *aligned faceset directory*.
    - We only return paths that already exist to avoid silently creating
      unexpected folders.
    """

    candidates = [
        workspace / 'pretrain_faces' / 'aligned',
        workspace / 'pretrain_faces',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _available_models(dfl_root: Path) -> list[str]:
    models_dir = dfl_root / 'models'
    if not models_dir.exists():
        return []
    # main.py 的 --model 取值为短名（AMP/Quick96/SAEHD/XSeg），不是 Model_*。
    out: list[str] = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith('Model_'):
            continue
        out.append(p.name.removeprefix('Model_'))
    out.sort()
    return out


def _choose_model(ctx: LaunchContext, default: str | None = 'SAEHD') -> str:
    models = _available_models(ctx.dfl_root)
    if not models:
        return _prompt('请输入模型类型（例如 SAEHD）', default or '')

    if default and default in models:
        default_model = default
    else:
        default_model = models[0]

    print('可用模型类型：')
    for i, name in enumerate(models, start=1):
        mark = ' (默认)' if name == default_model else ''
        print(f' {i:2d}. {name}{mark}')
    print()

    s = _prompt('请选择模型序号或直接输入模型名', default_model)
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(models):
            return models[idx - 1]
    return s


def _cmd_video_extract(ctx: LaunchContext) -> int:
    input_file = _prompt_path('输入视频文件（--input-file）', ctx.workspace / 'input.mp4', ensure=False)
    output_dir = _prompt_path('输出帧目录（--output-dir）', ctx.workspace / 'data_dst', ensure=True)
    output_ext = _prompt('输出图片格式（--output-ext，可留空）', '')
    fps_s = _prompt('每秒提取帧数（--fps，0=全帧率，可留空）', '')

    args = ['videoed', 'extract-video', '--input-file', str(input_file), '--output-dir', str(output_dir)]
    if output_ext:
        args += ['--output-ext', output_ext]
    if fps_s:
        args += ['--fps', fps_s]
    return _run_python_main(args, ctx)


def _cmd_bat_video_extract_data_src(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_src')
    return _run_python_main(
        [
            'videoed',
            'extract-video',
            '--input-file',
            str(ctx.workspace / 'data_src.*'),
            '--output-dir',
            str(ctx.workspace / 'data_src'),
        ],
        ctx,
    )


def _cmd_bat_video_extract_data_dst_full_fps(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    return _run_python_main(
        [
            'videoed',
            'extract-video',
            '--input-file',
            str(ctx.workspace / 'data_dst.*'),
            '--output-dir',
            str(ctx.workspace / 'data_dst'),
            '--fps',
            '0',
        ],
        ctx,
    )


def _cmd_bat_video_cut_drop_on_me(ctx: LaunchContext) -> int:
    input_file = _prompt_path('输入视频文件（--input-file）', ctx.workspace / 'input.mp4', ensure=False)
    return _run_python_main(['videoed', 'cut-video', '--input-file', str(input_file)], ctx)


def _cmd_bat_denoise_data_dst(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    return _run_python_main(
        ['videoed', 'denoise-image-sequence', '--input-dir', str(ctx.workspace / 'data_dst')],
        ctx,
    )


def _cmd_extract_faces(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}'
    output_dir = input_dir / 'aligned'

    input_dir = _prompt_path(f'输入目录（--input-dir，data_{which}）', input_dir, ensure=True)
    output_dir = _prompt_path(f'输出目录（--output-dir，data_{which}/aligned）', output_dir, ensure=True)
    detector = _prompt('detector（--detector：s3fd/manual，可留空走默认）', 's3fd')

    args = ['extract', '--input-dir', str(input_dir), '--output-dir', str(output_dir)]
    if detector:
        args += ['--detector', detector]
    return _run_python_main(args, ctx)


def _cmd_bat_extract_src_s3fd(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_src')
    _ensure_dir(ctx.workspace / 'data_src' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_src'),
            '--output-dir',
            str(ctx.workspace / 'data_src' / 'aligned'),
            '--detector',
            's3fd',
        ],
        ctx,
    )


def _cmd_bat_extract_src_manual(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_src')
    _ensure_dir(ctx.workspace / 'data_src' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_src'),
            '--output-dir',
            str(ctx.workspace / 'data_src' / 'aligned'),
            '--detector',
            'manual',
        ],
        ctx,
    )


def _cmd_bat_extract_dst_s3fd(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_dst'),
            '--output-dir',
            str(ctx.workspace / 'data_dst' / 'aligned'),
            '--detector',
            's3fd',
            '--max-faces-from-image',
            '0',
            '--output-debug',
        ],
        ctx,
    )


def _cmd_bat_extract_dst_manual(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_dst'),
            '--output-dir',
            str(ctx.workspace / 'data_dst' / 'aligned'),
            '--detector',
            'manual',
            '--max-faces-from-image',
            '0',
            '--output-debug',
        ],
        ctx,
    )


def _cmd_bat_extract_dst_manual_fix(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_dst'),
            '--output-dir',
            str(ctx.workspace / 'data_dst' / 'aligned'),
            '--output-debug',
            '--detector',
            's3fd',
            '--max-faces-from-image',
            '0',
            '--manual-fix',
        ],
        ctx,
    )


def _cmd_bat_extract_dst_manual_reextract_deleted_aligned_debug(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    return _run_python_main(
        [
            'extract',
            '--input-dir',
            str(ctx.workspace / 'data_dst'),
            '--output-dir',
            str(ctx.workspace / 'data_dst' / 'aligned'),
            '--detector',
            'manual',
            '--max-faces-from-image',
            '0',
            '--output-debug',
            '--manual-output-debug-fix',
        ],
        ctx,
    )


def _cmd_train(ctx: LaunchContext) -> int:
    src_aligned = _prompt_path('SRC faceset（--training-data-src-dir）', ctx.workspace / 'data_src' / 'aligned', ensure=True)
    dst_aligned = _prompt_path('DST faceset（--training-data-dst-dir）', ctx.workspace / 'data_dst' / 'aligned', ensure=True)
    model_dir = _prompt_path('模型目录（--model-dir）', ctx.workspace / 'model', ensure=True)
    model_name = _choose_model(ctx, default='SAEHD')

    want_pretrain = _prompt_yes_no('使用 pretrain 数据（需要 --pretraining-data-dir）', default=False)
    pretraining_dir: Path | None = None
    if want_pretrain:
        default_pretraining = _find_default_pretraining_dir(ctx.workspace)
        if default_pretraining is None:
            default_pretraining = ctx.workspace / 'pretrain_faces' / 'aligned'
        pretraining_dir = _prompt_path('pretrain faceset 目录（--pretraining-data-dir）', default_pretraining, ensure=False)

    silent_start = _prompt_yes_no('静默启动（--silent-start）', default=True)
    no_preview = _prompt_yes_no('禁用预览窗口（--no-preview）', default=False)

    args = [
        'train',
        '--training-data-src-dir', str(src_aligned),
        '--training-data-dst-dir', str(dst_aligned),
        '--model-dir', str(model_dir),
        '--model', model_name,
    ]
    if pretraining_dir is not None:
        args += ['--pretraining-data-dir', str(pretraining_dir)]
    if silent_start:
        args.append('--silent-start')
    if no_preview:
        args.append('--no-preview')

    return _run_python_main(args, ctx)


def _cmd_bat_train_model(ctx: LaunchContext, model: str, src_is_dst: bool = False) -> int:
    src_aligned = ctx.workspace / 'data_src' / 'aligned'
    dst_aligned = ctx.workspace / 'data_dst' / 'aligned'
    if src_is_dst:
        dst_aligned = src_aligned

    _ensure_dir(src_aligned)
    _ensure_dir(dst_aligned)
    _ensure_dir(ctx.workspace / 'model')

    pretraining_dir = _find_default_pretraining_dir(ctx.workspace)

    args = [
        'train',
        '--training-data-src-dir',
        str(src_aligned),
        '--training-data-dst-dir',
        str(dst_aligned),
        '--model-dir',
        str(ctx.workspace / 'model'),
        '--model',
        model,
    ]
    if pretraining_dir is not None:
        args += ['--pretraining-data-dir', str(pretraining_dir)]
    return _run_python_main(args, ctx)


def _cmd_merge(ctx: LaunchContext) -> int:
    input_dir = _prompt_path('输入帧目录（--input-dir）', ctx.workspace / 'data_dst', ensure=True)
    output_dir = _prompt_path('输出目录（--output-dir）', ctx.workspace / 'data_dst' / 'merged', ensure=True)
    output_mask_dir = _prompt_path('输出 mask 目录（--output-mask-dir）', ctx.workspace / 'data_dst' / 'merged_mask', ensure=True)
    aligned_dir = _prompt_path('aligned 目录（--aligned-dir）', ctx.workspace / 'data_dst' / 'aligned', ensure=False)
    model_dir = _prompt_path('模型目录（--model-dir）', ctx.workspace / 'model', ensure=True)
    model_name = _choose_model(ctx, default='SAEHD')

    args = [
        'merge',
        '--input-dir', str(input_dir),
        '--output-dir', str(output_dir),
        '--output-mask-dir', str(output_mask_dir),
        '--aligned-dir', str(aligned_dir),
        '--model-dir', str(model_dir),
        '--model', model_name,
    ]
    return _run_python_main(args, ctx)


def _cmd_bat_merge_model(ctx: LaunchContext, model: str) -> int:
    _ensure_dir(ctx.workspace / 'data_dst')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    _ensure_dir(ctx.workspace / 'data_dst' / 'merged')
    _ensure_dir(ctx.workspace / 'data_dst' / 'merged_mask')
    _ensure_dir(ctx.workspace / 'model')

    return _run_python_main(
        [
            'merge',
            '--input-dir',
            str(ctx.workspace / 'data_dst'),
            '--output-dir',
            str(ctx.workspace / 'data_dst' / 'merged'),
            '--output-mask-dir',
            str(ctx.workspace / 'data_dst' / 'merged_mask'),
            '--aligned-dir',
            str(ctx.workspace / 'data_dst' / 'aligned'),
            '--model-dir',
            str(ctx.workspace / 'model'),
            '--model',
            model,
        ],
        ctx,
    )


def _cmd_bat_export_dfm(ctx: LaunchContext, model: str) -> int:
    _ensure_dir(ctx.workspace / 'model')
    return _run_python_main(
        ['exportdfm', '--model-dir', str(ctx.workspace / 'model'), '--model', model],
        ctx,
    )


def _cmd_xseg_editor(ctx: LaunchContext) -> int:
    input_dir = _prompt_path('输入目录（--input-dir，通常是 aligned faces）', ctx.workspace / 'data_dst' / 'aligned', ensure=False)
    return _run_python_main(['xseg', 'editor', '--input-dir', str(input_dir)], ctx)


def _cmd_xseg_train(ctx: LaunchContext) -> int:
    src_aligned = _prompt_path('SRC faceset（--training-data-src-dir）', ctx.workspace / 'data_src' / 'aligned', ensure=True)
    dst_aligned = _prompt_path('DST faceset（--training-data-dst-dir）', ctx.workspace / 'data_dst' / 'aligned', ensure=True)
    model_dir = _prompt_path('模型目录（--model-dir）', ctx.workspace / 'model', ensure=True)
    want_pretrain = _prompt_yes_no('使用 pretrain 数据（需要 --pretraining-data-dir）', default=False)
    pretraining_dir: Path | None = None
    if want_pretrain:
        default_pretraining = _find_default_pretraining_dir(ctx.workspace)
        if default_pretraining is None:
            default_pretraining = ctx.workspace / 'pretrain_faces'
        pretraining_dir = _prompt_path('pretrain 目录（--pretraining-data-dir）', default_pretraining, ensure=False)
    silent_start = _prompt_yes_no('静默启动（--silent-start）', default=True)
    no_preview = _prompt_yes_no('禁用预览窗口（--no-preview）', default=False)

    args = [
        'xseg',
        'train',
        '--training-data-src-dir', str(src_aligned),
        '--training-data-dst-dir', str(dst_aligned),
        '--model-dir', str(model_dir),
    ]
    if pretraining_dir is not None:
        args += ['--pretraining-data-dir', str(pretraining_dir)]
    if silent_start:
        args.append('--silent-start')
    if no_preview:
        args.append('--no-preview')
    return _run_python_main(args, ctx)


def _cmd_bat_xseg_train(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace / 'data_src' / 'aligned')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    _ensure_dir(ctx.workspace / 'model')

    pretraining_dir = _find_default_pretraining_dir(ctx.workspace)

    args = [
        'xseg',
        'train',
        '--training-data-src-dir',
        str(ctx.workspace / 'data_src' / 'aligned'),
        '--training-data-dst-dir',
        str(ctx.workspace / 'data_dst' / 'aligned'),
        '--model-dir',
        str(ctx.workspace / 'model'),
    ]
    if pretraining_dir is not None:
        args += ['--pretraining-data-dir', str(pretraining_dir)]
    return _run_python_main(
        args,
        ctx,
    )


def _cmd_bat_xseg_apply_trained(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    _ensure_dir(ctx.workspace / 'model')
    return _run_python_main(
        ['xseg', 'apply', '--input-dir', str(input_dir), '--model-dir', str(ctx.workspace / 'model')],
        ctx,
    )


def _cmd_bat_xseg_remove(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['xseg', 'remove', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_xseg_remove_labels(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['xseg', 'remove_labels', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_xseg_fetch(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['xseg', 'fetch', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_xseg_editor(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['xseg', 'editor', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_util_add_landmarks_debug(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--add-landmarks-debug-images'], ctx)


def _cmd_bat_util_recover_original_filename(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--recover-original-aligned-filename'], ctx)


def _cmd_bat_util_pack_faceset(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--pack-faceset'], ctx)


def _cmd_bat_util_unpack_faceset(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--unpack-faceset'], ctx)


def _cmd_bat_util_save_metadata(ctx: LaunchContext) -> int:
    input_dir = ctx.workspace / 'data_src' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--save-faceset-metadata'], ctx)


def _cmd_bat_util_restore_metadata(ctx: LaunchContext) -> int:
    input_dir = ctx.workspace / 'data_src' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['util', '--input-dir', str(input_dir), '--restore-faceset-metadata'], ctx)


def _cmd_bat_faceset_resize(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['facesettool', 'resize', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_faceset_enhance_src(ctx: LaunchContext) -> int:
    input_dir = ctx.workspace / 'data_src' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['facesettool', 'enhance', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_sort(ctx: LaunchContext, which: str) -> int:
    if which not in ('src', 'dst'):
        raise ValueError(which)
    input_dir = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(input_dir)
    return _run_python_main(['sort', '--input-dir', str(input_dir)], ctx)


def _cmd_bat_view_aligned(ctx: LaunchContext, which: str) -> int:
    p = ctx.workspace / f'data_{which}' / 'aligned'
    _ensure_dir(p)
    _open_path(p)
    return 0


def _cmd_bat_view_aligned_debug(ctx: LaunchContext) -> int:
    p = ctx.workspace / 'data_dst' / 'aligned_debug'
    _ensure_dir(p)
    _open_path(p)
    return 0


def _cmd_bat_clear_workspace(ctx: LaunchContext) -> int:
    # 对齐原版 clear workspace.bat：清空 data_src/data_dst/model 并重建基础目录
    _ensure_dir(ctx.workspace)
    _rm_tree(ctx.workspace / 'data_src')
    _rm_tree(ctx.workspace / 'data_dst')
    _rm_tree(ctx.workspace / 'model')

    _ensure_dir(ctx.workspace / 'data_src' / 'aligned')
    _ensure_dir(ctx.workspace / 'data_dst' / 'aligned')
    _ensure_dir(ctx.workspace / 'model')
    print('DONE')
    return 0


def _cmd_bat_video_from_sequence_pair(
    ctx: LaunchContext,
    ext: str,
    merged_lossless: bool,
) -> int:
    merged_dir = ctx.workspace / 'data_dst' / 'merged'
    merged_mask_dir = ctx.workspace / 'data_dst' / 'merged_mask'
    _ensure_dir(merged_dir)
    _ensure_dir(merged_mask_dir)

    reference = str(ctx.workspace / 'data_dst.*')
    out_merged = str(ctx.workspace / f'result.{ext}')
    out_mask = str(ctx.workspace / f'result_mask.{ext}')

    args1 = [
        'videoed',
        'video-from-sequence',
        '--input-dir',
        str(merged_dir),
        '--output-file',
        out_merged,
        '--reference-file',
        reference,
        '--include-audio',
    ]
    if merged_lossless:
        args1.append('--lossless')

    rc = _run_python_main(args1, ctx)
    if rc != 0:
        return rc

    args2 = [
        'videoed',
        'video-from-sequence',
        '--input-dir',
        str(merged_mask_dir),
        '--output-file',
        out_mask,
        '--reference-file',
        reference,
        '--lossless',
    ]
    return _run_python_main(args2, ctx)


def _cmd_open_workspace(ctx: LaunchContext) -> int:
    _ensure_dir(ctx.workspace)
    _open_path(ctx.workspace)
    return 0


def _cmd_custom(ctx: LaunchContext) -> int:
    raw = _prompt('请输入 main.py 后面的参数（例如：extract --input-dir ...）', '')
    if not raw:
        return 0
    try:
        args = shlex.split(raw)
    except ValueError as e:
        print(f'命令解析失败：{e}')
        return 2
    return _run_python_main(args, ctx)


def _menu_items() -> list[MenuItem]:
    return [
        # 1) 初始化项目
        MenuItem('clear_workspace', '初始化项目：清空 workspace/data_src,data_dst,model', '1) 初始化项目', _cmd_bat_clear_workspace),

        # 2) 源视频转图片
        MenuItem('extract_images_src', '源视频转图片：extract images from video data_src', '2) 源视频转图片', _cmd_bat_video_extract_data_src),

        # 3) 目标视频转图片
        MenuItem('cut_video', '剪切视频（cut video）', '3) 目标视频转图片', _cmd_bat_video_cut_drop_on_me),
        MenuItem('extract_images_dst_fullfps', '目标视频转图片：extract images from video data_dst FULL FPS', '3) 目标视频转图片', _cmd_bat_video_extract_data_dst_full_fps),
        MenuItem('denoise_dst', '可选：去噪 data_dst 图像（denoise-image-sequence）', '3) 目标视频转图片', _cmd_bat_denoise_data_dst),

        # 4) 提取源头像
        MenuItem('src_extract', '提取源头像：data_src faceset extract（s3fd）', '4) 提取源头像', _cmd_bat_extract_src_s3fd),
        MenuItem('src_extract_manual', '提取源头像：手动（manual）', '4) 提取源头像', _cmd_bat_extract_src_manual),
        MenuItem('src_view_aligned', '查看源头像：data_src view aligned result', '4) 提取源头像', lambda ctx: _cmd_bat_view_aligned(ctx, 'src')),
        MenuItem('src_resize', '源头像：faceset resize', '4) 提取源头像', lambda ctx: _cmd_bat_faceset_resize(ctx, 'src')),
        MenuItem('src_add_landmarks_debug', '源头像：util add landmarks debug images', '4) 提取源头像', lambda ctx: _cmd_bat_util_add_landmarks_debug(ctx, 'src')),
        MenuItem('src_recover_filename', '源头像：util recover original filename', '4) 提取源头像', lambda ctx: _cmd_bat_util_recover_original_filename(ctx, 'src')),
        MenuItem('src_pack', '源头像：util faceset pack', '4) 提取源头像', lambda ctx: _cmd_bat_util_pack_faceset(ctx, 'src')),
        MenuItem('src_unpack', '源头像：util faceset unpack', '4) 提取源头像', lambda ctx: _cmd_bat_util_unpack_faceset(ctx, 'src')),
        MenuItem('src_sort', '源头像：data_src sort', '4) 提取源头像', lambda ctx: _cmd_bat_sort(ctx, 'src')),
        MenuItem('src_metadata_save', '源头像：faceset metadata save', '4) 提取源头像', _cmd_bat_util_save_metadata),
        MenuItem('src_metadata_restore', '源头像：faceset metadata restore', '4) 提取源头像', _cmd_bat_util_restore_metadata),
        MenuItem('src_enhance', '源头像：faceset enhance', '4) 提取源头像', _cmd_bat_faceset_enhance_src),

        # 5) 提取目标头像
        MenuItem('dst_extract', '目标头像提取：data_dst faceset extract（s3fd + debug）', '5) 提取目标头像', _cmd_bat_extract_dst_s3fd),
        MenuItem('dst_extract_manual', '目标头像提取：手动（manual + debug）', '5) 提取目标头像', _cmd_bat_extract_dst_manual),
        MenuItem('dst_extract_manual_fix', '目标头像提取：手动修复（s3fd + manual-fix）', '5) 提取目标头像', _cmd_bat_extract_dst_manual_fix),
        MenuItem('dst_manual_reextract_deleted_debug', '目标头像：删除 debug 后手动重提（manual-output-debug-fix）', '5) 提取目标头像', _cmd_bat_extract_dst_manual_reextract_deleted_aligned_debug),
        MenuItem('dst_view_aligned', '查看目标头像：data_dst view aligned results', '5) 提取目标头像', lambda ctx: _cmd_bat_view_aligned(ctx, 'dst')),
        MenuItem('dst_view_aligned_debug', '查看目标头像：data_dst view aligned_debug results', '5) 提取目标头像', _cmd_bat_view_aligned_debug),
        MenuItem('dst_resize', '目标头像：faceset resize', '5) 提取目标头像', lambda ctx: _cmd_bat_faceset_resize(ctx, 'dst')),
        MenuItem('dst_recover_filename', '目标头像：util recover original filename', '5) 提取目标头像', lambda ctx: _cmd_bat_util_recover_original_filename(ctx, 'dst')),
        MenuItem('dst_pack', '目标头像：util faceset pack', '5) 提取目标头像', lambda ctx: _cmd_bat_util_pack_faceset(ctx, 'dst')),
        MenuItem('dst_unpack', '目标头像：util faceset unpack', '5) 提取目标头像', lambda ctx: _cmd_bat_util_unpack_faceset(ctx, 'dst')),
        MenuItem('dst_sort', '目标头像：data_dst sort', '5) 提取目标头像', lambda ctx: _cmd_bat_sort(ctx, 'dst')),

        # 5.X) 遮罩处理（XSeg）
        MenuItem('mask_trained_src_apply', 'XSeg：源头像 trained mask apply（使用 workspace/model）', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_apply_trained(ctx, 'src')),
        MenuItem('mask_trained_src_remove', 'XSeg：源头像 trained mask remove', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_remove(ctx, 'src')),
        MenuItem('mask_src_fetch', 'XSeg：源头像 mask fetch', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_fetch(ctx, 'src')),
        MenuItem('mask_src_edit', 'XSeg：源头像 mask edit', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_editor(ctx, 'src')),
        MenuItem('mask_src_remove_labels', 'XSeg：源头像 mask remove labels', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_remove_labels(ctx, 'src')),
        MenuItem('mask_trained_dst_apply', 'XSeg：目标头像 trained mask apply（使用 workspace/model）', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_apply_trained(ctx, 'dst')),
        MenuItem('mask_trained_dst_remove', 'XSeg：目标头像 trained mask remove', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_remove(ctx, 'dst')),
        MenuItem('mask_dst_fetch', 'XSeg：目标头像 mask fetch', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_fetch(ctx, 'dst')),
        MenuItem('mask_dst_edit', 'XSeg：目标头像 mask edit', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_editor(ctx, 'dst')),
        MenuItem('mask_dst_remove_labels', 'XSeg：目标头像 mask remove labels', '5.X) 遮罩处理', lambda ctx: _cmd_bat_xseg_remove_labels(ctx, 'dst')),
        MenuItem('mask_train', 'XSeg：训练遮罩模型（xseg train）', '5.X) 遮罩处理', _cmd_bat_xseg_train),

        # 6) 训练模型
        MenuItem('train_amp', '训练 AMP 模型', '6) 训练模型', lambda ctx: _cmd_bat_train_model(ctx, 'AMP')),
        MenuItem('train_amp_srcsrc', '训练 AMP 模型（SRC-SRC）', '6) 训练模型', lambda ctx: _cmd_bat_train_model(ctx, 'AMP', src_is_dst=True)),
        MenuItem('train_quick96', '训练 Quick96 模型', '6) 训练模型', lambda ctx: _cmd_bat_train_model(ctx, 'Quick96')),
        MenuItem('train_saehd', '训练 SAEHD 模型', '6) 训练模型', lambda ctx: _cmd_bat_train_model(ctx, 'SAEHD')),
        MenuItem('export_amp_dfm', '导出 AMP 模型为 dfm', '6) 训练模型', lambda ctx: _cmd_bat_export_dfm(ctx, 'AMP')),
        MenuItem('export_saehd_dfm', '导出 SAEHD 模型为 dfm', '6) 训练模型', lambda ctx: _cmd_bat_export_dfm(ctx, 'SAEHD')),

        # 7) 应用模型
        MenuItem('merge_amp', '应用 AMP 模型（merge）', '7) 应用模型', lambda ctx: _cmd_bat_merge_model(ctx, 'AMP')),
        MenuItem('merge_quick96', '应用 Quick96 模型（merge）', '7) 应用模型', lambda ctx: _cmd_bat_merge_model(ctx, 'Quick96')),
        MenuItem('merge_saehd', '应用 SAEHD 模型（merge）', '7) 应用模型', lambda ctx: _cmd_bat_merge_model(ctx, 'SAEHD')),

        # 8) 合成视频
        MenuItem('merged_to_avi', '合成 AVI 视频（result.avi + result_mask.avi）', '8) 合成视频', lambda ctx: _cmd_bat_video_from_sequence_pair(ctx, 'avi', merged_lossless=False)),
        MenuItem('merged_to_mov_lossless', '合成 MOV 无损视频（result.mov + result_mask.mov）', '8) 合成视频', lambda ctx: _cmd_bat_video_from_sequence_pair(ctx, 'mov', merged_lossless=True)),
        MenuItem('merged_to_mp4', '合成 MP4 视频（result.mp4 + result_mask.mp4）', '8) 合成视频', lambda ctx: _cmd_bat_video_from_sequence_pair(ctx, 'mp4', merged_lossless=False)),
        MenuItem('merged_to_mp4_lossless', '合成 MP4 无损视频（result.mp4 + result_mask.mp4）', '8) 合成视频', lambda ctx: _cmd_bat_video_from_sequence_pair(ctx, 'mp4', merged_lossless=True)),

        # 其他 / 自定义
        MenuItem('help', '显示 main.py 帮助', '其他', lambda ctx: _run_python_main(['-h'], ctx)),
        MenuItem('video_extract_custom', '从视频提取图片帧（自定义参数）', '其他', _cmd_video_extract),
        MenuItem('extract_src_custom', '提取 SRC 人脸（自定义参数）', '其他', lambda ctx: _cmd_extract_faces(ctx, 'src')),
        MenuItem('extract_dst_custom', '提取 DST 人脸（自定义参数）', '其他', lambda ctx: _cmd_extract_faces(ctx, 'dst')),
        MenuItem('train_custom', '训练模型（自定义参数）', '其他', _cmd_train),
        MenuItem('merge_custom', '合成（自定义参数）', '其他', _cmd_merge),
        MenuItem('xseg_editor_custom', 'XSeg 编辑器（自定义参数）', '其他', _cmd_xseg_editor),
        MenuItem('xseg_train_custom', '训练 XSeg（自定义参数）', '其他', _cmd_xseg_train),
        MenuItem('open_workspace', '打开 workspace 目录', '其他', _cmd_open_workspace),
        MenuItem('custom', '自定义运行 main.py 命令', '其他', _cmd_custom),
    ]


def _resolve_ctx(workspace: str | None) -> LaunchContext:
    dfl_root = Path(__file__).resolve().parent

    ws = workspace or WORKSPACE_DIR or os.environ.get('DFL_WORKSPACE') or str(dfl_root / 'workspace')

    return LaunchContext(
        dfl_root=dfl_root,
        workspace=Path(ws).expanduser().resolve(),
        python_executable=sys.executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--list', action='store_true', help='列出可用菜单项并退出')
    parser.add_argument('--run', dest='run_key', default=None, help='直接运行指定菜单项（key）')
    parser.add_argument('--workspace', default=None, help='覆盖 WORKSPACE（默认：./workspace 或 $DFL_WORKSPACE）')

    args = parser.parse_args(argv)

    ctx = _resolve_ctx(args.workspace)
    _ensure_dir(ctx.workspace)
    items = _menu_items()
    by_key = {i.key.lower(): i for i in items}

    if args.list:
        for it in items:
            print(f'{it.key}: {it.title}')
        return 0

    if args.run_key:
        key = args.run_key.strip().lower()
        it = by_key.get(key)
        if it is None:
            print(f'未找到菜单项：{args.run_key}')
            print('可用项：')
            for x in items:
                print(f'  {x.key}: {x.title}')
            return 2
        return int(it.run(ctx))

    print('=' * 70)
    print('DeepFaceLab Torch 启动器（原版 bat 工作流移植）')
    print('=' * 70)
    print(f'WORKSPACE: {ctx.workspace}')
    print()

    while True:
        last_group: str | None = None
        for i, it in enumerate(items, start=1):
            if it.group != last_group:
                print(f'\n[{it.group}]')
                last_group = it.group
            print(f' {i:2d}. {it.title}  [{it.key}]')
        print('  0. 退出')
        print()

        s = input('请选择：').strip()
        if s == '0':
            return 0

        picked: MenuItem | None = None
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(items):
                picked = items[idx - 1]
        else:
            picked = by_key.get(s.lower())

        if picked is None:
            continue

        rc = int(picked.run(ctx))
        if rc != 0:
            print(f'命令返回码：{rc}')
        print()
        input('回车继续...')
        print()


if __name__ == '__main__':
    raise SystemExit(main())
