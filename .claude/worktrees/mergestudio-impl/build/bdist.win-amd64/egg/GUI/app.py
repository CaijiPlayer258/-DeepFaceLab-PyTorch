from __future__ import annotations

import ast
import base64
import heapq
import os
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import TclError
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from typing import Any

import cv2

from . import model_store


@dataclass
class UiState:
    workspace: Path
    model_dir: Path


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner.columnconfigure(1, weight=1)

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel support (mac/windows)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_inner_configure(self, _event: object | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, _event: object | None = None) -> None:
        # Make inner frame match canvas width
        self.canvas.itemconfigure(self.window_id, width=self.canvas.winfo_width())

    def _on_mousewheel(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
            return
        if getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")
            return
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")


class Tooltip:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._tw: tk.Toplevel | None = None
        self._after_id: str | None = None

    def attach(self, widget: tk.Widget, text_getter) -> None:  # type: ignore[no-untyped-def]
        widget.bind("<Enter>", lambda e: self._schedule_show(widget, text_getter))
        widget.bind("<Leave>", lambda e: self.hide())
        widget.bind("<ButtonPress>", lambda e: self.hide())

    def _schedule_show(self, widget: tk.Widget, text_getter) -> None:  # type: ignore[no-untyped-def]
        self.hide()
        self._after_id = self._root.after(400, lambda: self.show(widget, text_getter()))

    def show(self, widget: tk.Widget, text: str) -> None:
        if not text:
            return
        self.hide()

        tw = tk.Toplevel(self._root)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)

        lbl = ttk.Label(tw, text=text, justify="left")
        lbl.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        # Place near cursor
        x = self._root.winfo_pointerx() + 14
        y = self._root.winfo_pointery() + 14
        tw.wm_geometry(f"+{x}+{y}")

        self._tw = tw

    def hide(self) -> None:
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._tw is not None:
            try:
                self._tw.destroy()
            except Exception:
                pass
            self._tw = None


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("DeepFaceLab Torch - GUI")

        self._state = UiState(
            workspace=(Path(__file__).resolve().parents[1] / "workspace").resolve(),
            model_dir=(Path(__file__).resolve().parents[1] / "workspace" / "model").resolve(),
        )

        self._models: list[model_store.ModelInfo] = []
        self._current_model: model_store.ModelInfo | None = None
        self._current_options: dict[str, Any] = {}
        self._current_defaults: dict[str, Any] = {}
        self._current_choices: dict[str, list[Any]] = {}
        self._current_labels: dict[str, str] = {}
        self._current_help: dict[str, str] = {}
        self._current_types: dict[str, str] = {}
        self._current_order: list[str] = []

        self._param_vars: dict[str, tk.Variable] = {}
        self._param_choice_maps: dict[str, dict[str, Any]] = {}
        self._param_types: dict[str, str] = {}

        self._proc: subprocess.Popen[str] | None = None
        self._proc_reader: threading.Thread | None = None

        self._run_img_after_id: str | None = None
        self._preview_rgb: Any | None = None
        self._curve_rgb: Any | None = None
        self._preview_photo: tk.PhotoImage | None = None
        self._curve_photo: tk.PhotoImage | None = None

        self._history_image_paths: list[Path] = []

        self._build_layout()
        self._tooltip = Tooltip(self)
        self._apply_responsive_geometry()
        self._apply_default_pane_layout()
        self.status_var.set("就绪（点击‘刷新模型’加载列表）")

    # ---------------- geometry / layout ----------------
    def _apply_responsive_geometry(self) -> None:
        # Ensure default size is good and controls never get hidden.
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        width = min(1280, max(1100, screen_w - 80))
        height = min(820, max(700, screen_h - 120))

        min_w = min(1100, max(920, screen_w - 120))
        min_h = min(680, max(560, screen_h - 180))

        self.geometry(f"{width}x{height}")
        self.minsize(min_w, min_h)

    def _apply_default_pane_layout(self) -> None:
        """Make the model list narrower so preview area is larger by default."""
        paned = getattr(self, "_main_paned", None)
        if paned is None:
            return

        def _set() -> None:
            try:
                # Keep list usable but not dominant.
                target = 320
                w = int(self.winfo_width() or 0)
                if w > 0:
                    target = min(max(280, int(w * 0.28)), 380)
                paned.sashpos(0, target)
            except Exception:
                pass

        # Run after geometry/layout is realized.
        self.after(50, _set)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Top toolbar (always visible)
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(4, weight=1)

        ttk.Label(toolbar, text="Workspace:").grid(row=0, column=0, sticky="w")
        self.workspace_var = tk.StringVar(value=str(self._state.workspace))
        workspace_entry = ttk.Entry(toolbar, textvariable=self.workspace_var, width=44)
        workspace_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(toolbar, text="选择...", command=self._choose_workspace).grid(row=0, column=2)

        ttk.Label(toolbar, text="ModelDir:").grid(row=0, column=3, sticky="w", padx=(12, 0))
        self.model_dir_var = tk.StringVar(value=str(self._state.model_dir))
        model_entry = ttk.Entry(toolbar, textvariable=self.model_dir_var, width=44)
        model_entry.grid(row=0, column=4, sticky="ew", padx=(6, 6))
        ttk.Button(toolbar, text="选择...", command=self._choose_model_dir).grid(row=0, column=5)

        ttk.Button(toolbar, text="刷新模型", command=self._refresh_models_async).grid(row=0, column=6, padx=(12, 0))

        # Main area (paned) - left list, right parameters/log
        main = ttk.PanedWindow(self, orient="horizontal")
        self._main_paned = main
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        # Default ratio: keep list narrower, give more space to preview/run.
        main.add(left, weight=1)
        main.add(right, weight=6)
        try:
            main.paneconfigure(left, minsize=260)
            main.paneconfigure(right, minsize=680)
        except Exception:
            pass

        # Left: model list
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="模型列表", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

        # Keep model list compact; free space is used for history.
        self.model_tree = ttk.Treeview(left, columns=("name", "iter"), show="headings", height=8)
        self.model_tree.heading("name", text="模型")
        self.model_tree.heading("iter", text="iter")
        self.model_tree.column("name", width=260, anchor="w")
        self.model_tree.column("iter", width=70, anchor="e")
        self.model_tree.grid(row=1, column=0, sticky="nsew")

        model_scroll = ttk.Scrollbar(left, orient="vertical", command=self.model_tree.yview)
        model_scroll.grid(row=1, column=1, sticky="ns")
        self.model_tree.configure(yscrollcommand=model_scroll.set)
        self.model_tree.bind("<<TreeviewSelect>>", self._on_select_model)

        # Left bottom: preview history (show past previews)
        hist = ttk.Labelframe(left, text="预览历史", padding=6)
        hist.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        hist.columnconfigure(1, weight=1)
        hist.rowconfigure(1, weight=1)

        ttk.Label(hist, text="类型:").grid(row=0, column=0, sticky="w")
        self.history_kind_var = tk.StringVar(value="")
        self.history_kind_cb = ttk.Combobox(hist, textvariable=self.history_kind_var, state="readonly", width=18)
        self.history_kind_cb.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.history_kind_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_history_list())

        self.history_list = tk.Listbox(hist, height=8)
        self.history_list.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        hist_scroll = ttk.Scrollbar(hist, orient="vertical", command=self.history_list.yview)
        hist_scroll.grid(row=1, column=2, sticky="ns", pady=(6, 0))
        self.history_list.configure(yscrollcommand=hist_scroll.set)
        self.history_list.bind("<<ListboxSelect>>", self._on_select_history_item)

        # Right: vertical paned: (top) tabs + (bottom) log
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        vpaned = ttk.PanedWindow(right, orient="vertical")
        vpaned.grid(row=0, column=0, sticky="nsew")

        self.params_frame = ttk.Labelframe(vpaned, text="操作", padding=8)
        log_frame = ttk.Labelframe(vpaned, text="日志", padding=8)
        vpaned.add(self.params_frame, weight=3)
        vpaned.add(log_frame, weight=1)

        # Tabs: 参数 / 运行
        self.params_frame.columnconfigure(0, weight=1)
        self.params_frame.rowconfigure(0, weight=1)

        self.tabs = ttk.Notebook(self.params_frame)
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.tab_params = ttk.Frame(self.tabs, padding=8)
        self.tab_run = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.tab_params, text="参数")
        self.tabs.add(self.tab_run, text="运行")

        # --- 参数 tab ---
        self.tab_params.columnconfigure(0, weight=1)
        self.tab_params.rowconfigure(1, weight=1)

        params_toolbar = ttk.Frame(self.tab_params)
        params_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        params_toolbar.columnconfigure(3, weight=1)

        self.save_btn = ttk.Button(params_toolbar, text="写回模型(备份)", command=self._save_options_to_model)
        self.save_btn.grid(row=0, column=0)
        self.reload_btn = ttk.Button(params_toolbar, text="重新读取", command=self._reload_current_model)
        self.reload_btn.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(params_toolbar, text="参数说明").grid(
            row=0, column=2, padx=(12, 0), sticky="w"
        )

        self.form = ScrollableFrame(self.tab_params)
        self.form.grid(row=1, column=0, sticky="nsew")

        # --- 运行 tab ---
        self._build_run_tab()

        # Log: text with scrollbar, always visible
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=8, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_y.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_y.set)

        # Status bar (always visible)
        status = ttk.Frame(self, padding=(10, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)

        ttk.Label(status, text="状态:").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

    def _build_run_tab(self) -> None:
        self.tab_run.columnconfigure(0, weight=1)
        self.tab_run.rowconfigure(2, weight=1)

        # --- 顶部：路径 + 控制按钮 ---
        top = ttk.Frame(self.tab_run)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="SRC(aligned):").grid(row=0, column=0, sticky="w")
        self.src_aligned_var = tk.StringVar(value=str((self._state.workspace / "data_src" / "aligned").resolve()))
        ttk.Entry(top, textvariable=self.src_aligned_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="选择...", command=lambda: self._choose_dir_into(self.src_aligned_var)).grid(row=0, column=2)

        ttk.Label(top, text="DST(aligned):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dst_aligned_var = tk.StringVar(value=str((self._state.workspace / "data_dst" / "aligned").resolve()))
        ttk.Entry(top, textvariable=self.dst_aligned_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Button(top, text="选择...", command=lambda: self._choose_dir_into(self.dst_aligned_var)).grid(row=1, column=2, pady=(6, 0))

        opts = ttk.Frame(top)
        opts.grid(row=0, column=3, rowspan=2, columnspan=2, sticky="e", padx=(12, 0))

        self.silent_start_var = tk.BooleanVar(value=True)
        self.no_preview_var = tk.BooleanVar(value=False)
        self.auto_refresh_imgs_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(opts, text="静默启动", variable=self.silent_start_var, command=self._refresh_command_preview).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="禁用预览窗口", variable=self.no_preview_var, command=self._refresh_command_preview).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(opts, text="自动刷新预览/曲线", variable=self.auto_refresh_imgs_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # --- 第二行：按钮条 ---
        btnbar = ttk.Frame(self.tab_run)
        btnbar.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        btnbar.columnconfigure(9, weight=1)

        self.run_save_btn = ttk.Button(btnbar, text="写回模型(备份)", command=self._save_options_to_model)
        self.run_save_btn.grid(row=0, column=0)
        self.run_reload_btn = ttk.Button(btnbar, text="重新读取", command=self._reload_current_model)
        self.run_reload_btn.grid(row=0, column=1, padx=(8, 0))
        self.refresh_imgs_btn = ttk.Button(btnbar, text="刷新预览/曲线", command=self._refresh_run_images)
        self.refresh_imgs_btn.grid(row=0, column=2, padx=(16, 0))

        self.copy_cmd_btn = ttk.Button(btnbar, text="复制命令", command=self._copy_command)
        self.copy_cmd_btn.grid(row=0, column=3, padx=(16, 0))
        self.start_btn = ttk.Button(btnbar, text="开始训练", command=self._start_train)
        self.start_btn.grid(row=0, column=4, padx=(8, 0))
        self.stop_btn = ttk.Button(btnbar, text="停止", command=self._stop_process, state="disabled")
        self.stop_btn.grid(row=0, column=5, padx=(8, 0))

        # --- 中部：左右分栏（预览 / 曲线） ---
        mid = ttk.PanedWindow(self.tab_run, orient="horizontal")
        mid.grid(row=2, column=0, sticky="nsew")

        preview_box = ttk.Labelframe(mid, text="预览", padding=8)
        curve_box = ttk.Labelframe(mid, text="曲线", padding=8)
        mid.add(preview_box, weight=3)
        mid.add(curve_box, weight=2)

        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        curve_box.columnconfigure(0, weight=1)
        curve_box.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_box, text="(未加载)", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.curve_label = ttk.Label(curve_box, text="(未加载)", anchor="center")
        self.curve_label.grid(row=0, column=0, sticky="nsew")

        # Resize -> re-render
        self.preview_label.bind("<Configure>", lambda _e: self._rerender_loaded_images())
        self.curve_label.bind("<Configure>", lambda _e: self._rerender_loaded_images())

        # --- 底部：命令预览 ---
        bottom = ttk.Frame(self.tab_run)
        bottom.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)

        ttk.Label(bottom, text="命令预览（可复制）：").grid(row=0, column=0, sticky="w")
        self.cmd_text = tk.Text(bottom, height=4, wrap="word")
        self.cmd_text.grid(row=1, column=0, sticky="nsew")
        self.cmd_text.configure(state="disabled")

        # Update when paths change
        self.src_aligned_var.trace_add("write", lambda *_: self._refresh_command_preview())
        self.dst_aligned_var.trace_add("write", lambda *_: self._refresh_command_preview())
        self.model_dir_var.trace_add("write", lambda *_: self._refresh_command_preview())
        self._refresh_command_preview()
        self._refresh_run_images()

    def _choose_dir_into(self, var: tk.StringVar) -> None:
        p = filedialog.askdirectory(initialdir=var.get() or os.getcwd())
        if not p:
            return
        var.set(str(Path(p).expanduser().resolve()))

    # ---------------- actions ----------------
    def _log(self, s: str) -> None:
        self.log_text.insert("end", s + "\n")
        self.log_text.see("end")

    def _choose_workspace(self) -> None:
        p = filedialog.askdirectory(initialdir=self.workspace_var.get() or os.getcwd())
        if not p:
            return
        self._state.workspace = Path(p).expanduser().resolve()
        self.workspace_var.set(str(self._state.workspace))
        # If model_dir is under workspace/model, update it automatically.
        guess_model = self._state.workspace / "model"
        if guess_model.exists():
            self._state.model_dir = guess_model
            self.model_dir_var.set(str(self._state.model_dir))
        self._refresh_models_async()

    def _choose_model_dir(self) -> None:
        p = filedialog.askdirectory(initialdir=self.model_dir_var.get() or os.getcwd())
        if not p:
            return
        self._state.model_dir = Path(p).expanduser().resolve()
        self.model_dir_var.set(str(self._state.model_dir))
        self._refresh_models_async()

    def _refresh_models_async(self) -> None:
        # Avoid freezing UI when scanning lots of files.
        self.status_var.set("正在扫描模型...")
        self._log("[GUI] 扫描模型目录...")

        def worker() -> None:
            try:
                model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
                models = model_store.scan_models(model_dir)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("扫描失败", str(e)))
                models = []

            self.after(0, lambda: self._apply_models(models))

        threading.Thread(target=worker, name="scan-models", daemon=True).start()

    def _apply_models(self, models: list[model_store.ModelInfo]) -> None:
        self._models = models

        # Clear list
        for item in self.model_tree.get_children(""):
            self.model_tree.delete(item)

        for m in models:
            self.model_tree.insert("", "end", iid=m.data_path.as_posix(), values=(m.display_name, m.iter))

        self.status_var.set(f"模型数：{len(models)}")
        self._log(f"[GUI] 发现模型：{len(models)}")

        # Auto-select first model if any
        if models:
            first_id = models[0].data_path.as_posix()
            try:
                self.model_tree.selection_set(first_id)
                self.model_tree.focus(first_id)
                self.model_tree.see(first_id)
                self._load_model_by_path(Path(first_id))
            except Exception:
                pass
        else:
            self._current_model = None
            self._render_param_form({}, {}, {}, {}, [])
            self._refresh_history_kinds()
            self._refresh_history_list()

    def _on_select_model(self, _event: object | None = None) -> None:
        sel = self.model_tree.selection()
        if not sel:
            return
        p = Path(sel[0])
        self._load_model_by_path(p)

    def _load_model_by_path(self, data_path: Path) -> None:
        try:
            m = next((x for x in self._models if x.data_path.resolve() == data_path.resolve()), None)
            if m is None:
                return
            self._current_model = m
            model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
            self._current_options = model_store.read_options(m.data_path)
            self._current_defaults = model_store.read_default_options(model_dir, m.model_class)
            self._current_choices = model_store.read_model_choices(m.model_class)
            self._current_labels = model_store.read_model_labels(m.model_class)
            self._current_help = model_store.read_model_help(m.model_class)
            self._current_types = model_store.read_model_types(m.model_class)
            self._current_order = model_store.read_model_order(m.model_class)
            self._render_param_form(
                self._current_options,
                self._current_defaults,
                self._current_choices,
                self._current_help,
                self._current_order,
            )
            self.status_var.set(f"已加载：{m.display_name}  iter={m.iter}")
            self._refresh_command_preview()
            self._refresh_run_images()
            self._refresh_history_kinds()
            self._refresh_history_list()
        except Exception as e:
            messagebox.showerror("读取模型失败", str(e))

    def _build_train_cmd(self) -> list[str] | None:
        if self._current_model is None:
            return None

        src = Path(self.src_aligned_var.get()).expanduser()
        dst = Path(self.dst_aligned_var.get()).expanduser()
        model_dir = Path(self.model_dir_var.get()).expanduser()
        if not src.exists() or not dst.exists() or not model_dir.exists():
            return None

        m = self._current_model
        # main.py train expects bare class names (choices: AMP, Quick96, SAEHD, XSeg)
        model_class_arg = m.model_class
        force_name = m.display_name

        cmd = [sys.executable, "-u", "main.py", "train",
               "--training-data-src-dir", str(src),
               "--training-data-dst-dir", str(dst),
               "--model-dir", str(model_dir),
               "--model", model_class_arg,
               "--force-model-name", force_name]

        if self.silent_start_var.get():
            cmd.append("--silent-start")
        if self.no_preview_var.get():
            cmd.append("--no-preview")

        return cmd

    def _refresh_command_preview(self) -> None:
        cmd = self._build_train_cmd()
        if cmd is None:
            text = "(请选择模型，并确认 SRC/DST aligned 与 model_dir 路径存在)"
        else:
            text = " ".join(shlex.quote(x) for x in cmd)

        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", text)
        self.cmd_text.configure(state="disabled")

        can_run = cmd is not None and self._proc is None
        self.start_btn.configure(state="normal" if can_run else "disabled")

    def _copy_command(self) -> None:
        try:
            txt = self.cmd_text.get("1.0", "end").strip()
            if not txt:
                return
            self.clipboard_clear()
            self.clipboard_append(txt)
            self._log("[GUI] 已复制命令到剪贴板")
        except TclError:
            pass

    def _start_train(self) -> None:
        if self._proc is not None:
            return
        cmd = self._build_train_cmd()
        if cmd is None:
            messagebox.showerror("无法启动", "请先选择模型，并确认 SRC/DST aligned 与 model_dir 路径存在。")
            return

        self._log("[RUN] 启动训练...")
        self._log("[RUN] " + " ".join(shlex.quote(x) for x in cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self._proc = None
            messagebox.showerror("启动失败", str(e))
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("训练中...")

        if self.auto_refresh_imgs_var.get():
            self._schedule_auto_refresh_images()

        t = threading.Thread(target=self._read_proc_output, name="proc-reader", daemon=True)
        self._proc_reader = t
        t.start()

    def _read_proc_output(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                s = line.rstrip("\n")
                if s:
                    self.after(0, lambda ss=s: self._log(ss))
        except Exception as e:
            self.after(0, lambda: self._log(f"[RUN] 输出读取失败: {e}"))
        finally:
            code = proc.poll()
            if code is None:
                code = proc.wait()
            self.after(0, lambda: self._on_proc_exit(int(code)))

    def _on_proc_exit(self, code: int) -> None:
        self._log(f"[RUN] 进程结束，exit={code}")
        self._proc = None
        self._proc_reader = None
        self.stop_btn.configure(state="disabled")
        self.status_var.set("就绪")
        self._stop_auto_refresh_images()
        self._refresh_run_images()
        self._refresh_command_preview()

    # ---------------- preview / curve ----------------
    def _stop_auto_refresh_images(self) -> None:
        if self._run_img_after_id is not None:
            try:
                self.after_cancel(self._run_img_after_id)
            except Exception:
                pass
            self._run_img_after_id = None

    def _schedule_auto_refresh_images(self) -> None:
        # Refresh while process is alive.
        self._stop_auto_refresh_images()

        def tick() -> None:
            if self._proc is None:
                self._run_img_after_id = None
                return
            try:
                self._refresh_run_images()
            finally:
                self._run_img_after_id = self.after(1500, tick)

        self._run_img_after_id = self.after(300, tick)

    def _rerender_loaded_images(self) -> None:
        # When label size changes, re-render cached RGB arrays.
        if self._preview_rgb is not None:
            self._preview_photo = self._render_rgb_into_label(self.preview_label, self._preview_rgb)
        if self._curve_rgb is not None:
            self._curve_photo = self._render_rgb_into_label(self.curve_label, self._curve_rgb)

    def _refresh_run_images(self) -> None:
        if self._current_model is None:
            self.preview_label.configure(text="(请选择模型)", image="")
            self.curve_label.configure(text="(请选择模型)", image="")
            self._preview_rgb = None
            self._curve_rgb = None
            self._preview_photo = None
            self._curve_photo = None
            return

        model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
        model_name = self._current_model.display_name
        hist_root = model_dir / f"{model_name}_history"

        last_files: list[Path] = []
        try:
            if hist_root.exists():
                last_files = [p for p in hist_root.glob("*/_last.jpg") if p.is_file()]
        except Exception:
            last_files = []

        if not last_files:
            # Also try legacy in model_dir root (colab mode saves preview_*.jpg)
            try:
                last_files = [p for p in model_dir.glob("preview_*.jpg") if p.is_file()]
            except Exception:
                last_files = []

        if not last_files:
            self.preview_label.configure(text="(未找到预览文件：请在参数里开启 write_preview_history)", image="")
            self.curve_label.configure(text="(未找到曲线文件)", image="")
            self._preview_rgb = None
            self._curve_rgb = None
            self._preview_photo = None
            self._curve_photo = None
            return

        last_path = max(last_files, key=lambda p: p.stat().st_mtime)
        img_bgr = cv2.imread(str(last_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            self.preview_label.configure(text=f"(读取失败: {last_path.name})", image="")
            self.curve_label.configure(text=f"(读取失败: {last_path.name})", image="")
            self._preview_rgb = None
            self._curve_rgb = None
            self._preview_photo = None
            self._curve_photo = None
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h = int(img_rgb.shape[0])

        # According to ModelBase.get_loss_history_preview(): loss graph height is always 100.
        lh_height = 100
        if h <= lh_height + 1:
            curve_rgb = img_rgb
            preview_rgb = img_rgb
        else:
            curve_rgb = img_rgb[:lh_height, :, :]
            preview_rgb = img_rgb[lh_height:, :, :]

        self._preview_rgb = preview_rgb
        self._curve_rgb = curve_rgb
        self._preview_photo = self._render_rgb_into_label(self.preview_label, preview_rgb)
        self._curve_photo = self._render_rgb_into_label(self.curve_label, curve_rgb)

        # Sync history panel (new images may have been written).
        self._refresh_history_kinds()
        self._refresh_history_list()

    def _get_history_root(self) -> Path | None:
        if self._current_model is None:
            return None
        model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
        model_name = self._current_model.display_name
        root = model_dir / f"{model_name}_history"
        return root if root.exists() else None

    def _refresh_history_kinds(self) -> None:
        if not hasattr(self, "history_kind_cb"):
            return
        root = self._get_history_root()
        kinds: list[str] = []
        if root is not None:
            try:
                kinds = sorted([p.name for p in root.iterdir() if p.is_dir()])
            except Exception:
                kinds = []

        cur = self.history_kind_var.get()
        self.history_kind_cb["values"] = kinds
        if kinds and cur not in kinds:
            self.history_kind_var.set(kinds[0])
        if not kinds:
            self.history_kind_var.set("")

    def _refresh_history_list(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.configure(state="normal")
        self.history_list.delete(0, "end")
        self._history_image_paths = []

        if self._current_model is None:
            self.history_list.insert("end", "(请选择模型)")
            self.history_list.configure(state="disabled")
            return

        root = self._get_history_root()
        kind = self.history_kind_var.get().strip()
        if root is None or not kind:
            self.history_list.insert("end", "(无历史)")
            self.history_list.configure(state="disabled")
            return

        folder = root / kind
        if not folder.exists():
            self.history_list.insert("end", "(无历史)")
            self.history_list.configure(state="disabled")
            return

        # Only keep newest N by filename (zero-padded, lexicographic works).
        N = 80
        try:
            entries = []
            for de in os.scandir(folder):
                if not de.is_file():
                    continue
                name = de.name
                if not name.lower().endswith(".jpg"):
                    continue
                if name.lower() == "_last.jpg":
                    continue
                entries.append(de)

            newest = heapq.nlargest(N, entries, key=lambda e: e.name)
            newest.sort(key=lambda e: e.name, reverse=True)
        except Exception:
            newest = []

        if not newest:
            self.history_list.insert("end", "(无历史)")
            self.history_list.configure(state="disabled")
            return

        for de in newest:
            p = Path(de.path)
            self._history_image_paths.append(p)
            self.history_list.insert("end", p.stem)

    def _on_select_history_item(self, _event: object | None = None) -> None:
        if not hasattr(self, "history_list"):
            return
        sel = self.history_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._history_image_paths):
            return

        p = self._history_image_paths[idx]
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        lh_height = 100
        h = int(img_rgb.shape[0])
        if h <= lh_height + 1:
            curve_rgb = img_rgb
            preview_rgb = img_rgb
        else:
            curve_rgb = img_rgb[:lh_height, :, :]
            preview_rgb = img_rgb[lh_height:, :, :]

        self._preview_rgb = preview_rgb
        self._curve_rgb = curve_rgb
        self._preview_photo = self._render_rgb_into_label(self.preview_label, preview_rgb)
        self._curve_photo = self._render_rgb_into_label(self.curve_label, curve_rgb)

    @staticmethod
    def _render_rgb_into_label(label: ttk.Label, rgb) -> tk.PhotoImage | None:  # type: ignore[no-untyped-def]
        try:
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
        except Exception:
            label.configure(text="(无效图像)", image="")
            return None

        target_w = max(1, int(label.winfo_width() or 1))
        target_h = max(1, int(label.winfo_height() or 1))
        # First layout pass returns 1x1; pick a sane fallback.
        if target_w <= 5 or target_h <= 5:
            target_w, target_h = 640, 360

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        try:
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = rgb

        # Build a binary PPM (P6) and feed to PhotoImage via base64.
        rh, rw = int(resized.shape[0]), int(resized.shape[1])
        ppm_header = f"P6 {rw} {rh} 255\n".encode("ascii")
        ppm = ppm_header + resized.tobytes(order="C")
        b64 = base64.b64encode(ppm)

        try:
            photo = tk.PhotoImage(data=b64, format="PPM")
            label.configure(image=photo, text="")
            return photo
        except Exception:
            label.configure(text="(渲染失败：请确认 Tk 支持 PPM)", image="")
            return None

    def _stop_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._log("[RUN] 正在停止...")
        try:
            proc.terminate()
        except Exception:
            pass

        def kill_later() -> None:
            p2 = self._proc
            if p2 is None:
                return
            if p2.poll() is None:
                try:
                    p2.kill()
                except Exception:
                    pass

        self.after(2000, kill_later)

    def _render_param_form(
        self,
        options: dict[str, Any],
        defaults: dict[str, Any],
        choices: dict[str, list[Any]],
        help_text: dict[str, str],
        order: list[str],
    ) -> None:
        # Clear form
        for child in list(self.form.inner.winfo_children()):
            child.destroy()
        self._param_vars.clear()
        self._param_choice_maps.clear()
        self._param_types.clear()

        all_keys = set(options.keys()) | set(defaults.keys())

        # Prefer model-source order as the fixed schema.
        keys: list[str] = []
        if order:
            for k in order:
                if k in all_keys and k not in keys:
                    keys.append(k)
            # Keep any unexpected keys (older model files) at bottom.
            for k in sorted(all_keys):
                if k not in keys:
                    keys.append(k)
        else:
            keys = sorted(all_keys)

        # 2-column table: param / value. Help is shown as tooltip on hover.
        self.form.inner.grid_columnconfigure(0, weight=0)
        self.form.inner.grid_columnconfigure(1, weight=1)

        ttk.Label(self.form.inner, text="参数", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self.form.inner, text="值", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, sticky="w")

        r = 1
        for key in keys:
            raw_default = defaults.get(key, "")
            default = self._safe_default_value(raw_default)
            cur = options.get(key, default)

            label = self._current_labels.get(key, "").strip()
            left_text = label if label else key
            name_lbl = ttk.Label(self.form.inner, text=left_text)
            name_lbl.grid(row=r, column=0, sticky="nw", padx=(0, 12), pady=3)

            # Determine expected type
            expected_type = self._current_types.get(key) or ("bool" if isinstance(default, bool) else "str")
            self._param_types[key] = expected_type

            if key in choices and choices[key]:
                var = tk.StringVar(value=str(cur))
                display_to_val = {str(v): v for v in choices[key]}
                self._param_choice_maps[key] = display_to_val
                cb = ttk.Combobox(self.form.inner, textvariable=var, values=list(display_to_val.keys()), state="readonly", width=22)
                cb.grid(row=r, column=1, sticky="nw", pady=3)
                self._param_vars[key] = var
            elif expected_type == "bool" or isinstance(cur, bool) or isinstance(default, bool):
                varb = tk.BooleanVar(value=bool(cur))
                chk = ttk.Checkbutton(self.form.inner, variable=varb)
                chk.grid(row=r, column=1, sticky="nw", pady=3)
                self._param_vars[key] = varb
            else:
                var = tk.StringVar(value=self._format_edit_prefill(cur))
                ent = ttk.Entry(self.form.inner, textvariable=var, width=24)
                ent.grid(row=r, column=1, sticky="nw", pady=3)
                self._param_vars[key] = var

            # Tooltip text: help + choices.
            def _tip_text_for_key(k: str = key, lbl: str = label) -> str:
                msg = help_text.get(k, "").strip()
                if not msg and lbl:
                    msg = lbl
                lines: list[str] = []
                if msg:
                    lines.append(msg)
                if k in choices and choices[k]:
                    opts = ", ".join(str(x) for x in choices[k])
                    lines.append(f"可选: {opts}")
                return "\n".join(lines).strip()

            self._tooltip.attach(name_lbl, _tip_text_for_key)

            r += 1

        self.save_btn.configure(state="normal" if self._current_model else "disabled")
        self.reload_btn.configure(state="normal" if self._current_model else "disabled")

    @staticmethod
    def _to_plain_scalar(v: Any) -> Any:
        """Best-effort convert numpy/torch scalars to plain Python types for display."""
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        item = getattr(v, "item", None)
        if callable(item):
            try:
                vv = item()
                if vv is None or isinstance(vv, (str, int, float, bool)):
                    return vv
            except Exception:
                pass
        return v

    @classmethod
    def _is_probably_expression(cls, s: str) -> bool:
        # Generated defaults may contain expressions; we don't want users to "calculate" them.
        s2 = s.strip()
        if s2 == "":
            return False
        markers = ("self.", "np.", "torch.", " if ", " else ", "//", "==", "!=", ":", "default_", "fast_smoke")
        if any(m in s2 for m in markers):
            return True
        # Heuristic: expressions often contain parentheses.
        if "(" in s2 or ")" in s2:
            return True
        return False

    @classmethod
    def _safe_default_value(cls, v: Any) -> Any:
        # Hide expression-like defaults; treat as not set.
        if isinstance(v, str) and cls._is_probably_expression(v):
            return ""
        return cls._to_plain_scalar(v)

    @staticmethod
    def _format_edit_prefill(v: Any) -> str:
        v = MainWindow._to_plain_scalar(v)
        if v == "":
            return ""
        if v is None:
            return "None"
        if isinstance(v, str):
            # Let user decide if they want quotes; keep raw string as-is.
            return v
        try:
            return str(v)
        except Exception:
            return ""

    @staticmethod
    def _format_choice_value(v: Any) -> str:
        if v is None:
            return "None"
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v)

    @staticmethod
    def _parse_value(text: str) -> Any:
        s = text.strip()
        if s == "":
            return ""
        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        if low in {"none", "null"}:
            return None
        # Fast numeric path
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        try:
            if any(c in s for c in (".", "e", "E")):
                return float(s)
        except Exception:
            pass

        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    def _reload_current_model(self) -> None:
        if self._current_model is None:
            return
        self._load_model_by_path(self._current_model.data_path)

    def _save_options_to_model(self) -> None:
        if self._current_model is None:
            return
        m = self._current_model

        ok = messagebox.askokcancel(
            "确认写回",
            "将把当前表单参数写回模型 data.dat。\n\n将自动创建 .bak 备份。\n\n确定写回吗？",
        )
        if not ok:
            return

        # Build new options from form vars.
        new_opts: dict[str, Any] = {}
        bad: list[str] = []
        for key, var in self._param_vars.items():
            expected = self._param_types.get(key, "any")

            if isinstance(var, tk.BooleanVar):
                new_opts[key] = bool(var.get())
                continue

            raw = str(var.get()).strip()

            if key in self._param_choice_maps:
                mapp = self._param_choice_maps[key]
                if raw in mapp:
                    new_opts[key] = mapp[raw]
                else:
                    new_opts[key] = raw
                continue

            try:
                if expected == "int":
                    new_opts[key] = int(raw) if raw != "" else 0
                elif expected == "float":
                    new_opts[key] = float(raw) if raw != "" else 0.0
                elif expected == "str":
                    new_opts[key] = raw
                else:
                    new_opts[key] = self._parse_value(raw)
            except Exception:
                bad.append(f"{key}: {raw}")

        if bad:
            messagebox.showerror("参数格式错误", "以下参数格式不正确，请修正后再写回：\n\n" + "\n".join(bad[:30]))
            return

        try:
            backup = model_store.write_options(m.data_path, new_opts, make_backup=True)
        except Exception as e:
            messagebox.showerror("写回失败", str(e))
            return

        self.status_var.set("写回成功（已备份）")
        if backup is not None:
            self._log(f"[GUI] 写回成功，备份：{backup}")
        else:
            self._log("[GUI] 写回成功")
        # Reload to reflect persisted state
        self._reload_current_model()

    @staticmethod
    def _format_value(v: Any) -> str:
        v = MainWindow._to_plain_scalar(v)
        try:
            if isinstance(v, (dict, list, tuple)):
                s = str(v)
            else:
                s = "" if v is None else str(v)
        except Exception:
            s = "<unrepr>"
        if len(s) > 260:
            return s[:260] + "..."
        return s


def main() -> int:
    try:
        app = MainWindow()
        app.mainloop()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
