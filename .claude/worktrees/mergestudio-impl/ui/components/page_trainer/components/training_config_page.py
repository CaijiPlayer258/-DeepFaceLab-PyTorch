"""
训练配置子页面 - 点击训练按钮时弹出（完整配置）
"""
import subprocess
import sys
import platform
from pathlib import Path
from PyQt5.QtWidgets import QBoxLayout, QHBoxLayout, QVBoxLayout, QWidget
from siui.components.page.child_page import SiChildPage
from siui.components import SiTitledWidgetGroup, SiPushButton, SiSwitch
from siui.components.editbox import SiLabeledLineEdit
from siui.components.option_card import SiOptionCardLinear
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.widgets import SiCheckBox
from siui.components.widgets.label import SiLabel, SiSvgLabel
from siui.core import SiGlobal


def safe_get_icon(name):
    """安全获取图标"""
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


def _detect_available_devices():
    """检测可用的 GPU（通过 nvidia-smi）和 CPU 设备"""
    devices = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    parts = line.split(', ', 1)
                    if len(parts) == 2:
                        devices.append({'type': 'gpu', 'index': parts[0].strip(), 'name': parts[1].strip()})
    except Exception:
        pass
    try:
        cpu_name = platform.processor()
        # Windows 下 platform.processor() 返回的是通用标识，用 wmic 获取实际名称
        if platform.system() == 'Windows':
            try:
                r = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    for line in r.stdout.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('Name'):
                            cpu_name = line
                            break
            except Exception:
                pass
        if not cpu_name:
            cpu_name = platform.machine()
        devices.append({'type': 'cpu', 'index': '', 'name': cpu_name or 'CPU'})
    except Exception:
        devices.append({'type': 'cpu', 'index': '', 'name': 'CPU'})
    return devices


class TrainingConfigChildPage(SiChildPage):
    """训练配置子页面 - 点击训练按钮时弹出（完整配置）"""
    def __init__(self, parent=None, model_name="model_001", model_info=None, model_dir="", webui_port="6789", webui_password="caiji"):
        super().__init__(parent)
        self._model_name = model_name
        self._model_dir = model_dir
        self._webui_port = webui_port
        self._webui_password = webui_password
        self.model_info = model_info or {}
        self.config_data = self._build_config_data()

        self.view().setMinimumWidth(900)
        self.content().setTitle(f"训练配置 - {model_name}")
        self.content().setPadding(48)
        
        # 页面内容
        self.titled_widget_group = SiTitledWidgetGroup(self)
        
        with self.titled_widget_group as group:
            group.addTitle("模型架构参数")
            
            # 模型名称
            self.model_name_card = SiOptionCardLinear(self)
            self.model_name_card.setTitle("模型名称", "设置模型的名称")
            self.model_name_card.load(safe_get_icon("ic_fluent_textbox_filled"))
            self.model_name_input = SiLabeledLineEdit(self.model_name_card)
            self.model_name_input.setTitle("字符串")
            self.model_name_input.setText(model_name)
            self.model_name_input.setFixedHeight(48)
            self.model_name_input.resize(150, 48)
            self.model_name_input.textChanged.connect(lambda: self.update_config('model_name'))
            self.model_name_card.addWidget(self.model_name_input)
            self.model_name_card.adjustSize()
            
            # 分辨率
            self.resolution_card = SiOptionCardLinear(self)
            self.resolution_card.setTitle("分辨率", "模型输入输出的图像分辨率")
            self.resolution_card.load(safe_get_icon("ic_fluent_scan_filled"))
            self.resolution_input = SiLabeledLineEdit(self.resolution_card)
            self.resolution_input.setTitle("整数")
            self.resolution_input.setText("256")
            self.resolution_input.setFixedHeight(48)
            self.resolution_input.resize(150, 48)
            self.resolution_input.textChanged.connect(lambda: self.update_config('resolution'))
            self.resolution_card.addWidget(self.resolution_input)
            self.resolution_card.adjustSize()
            
            # archi - 模型架构（不可编辑）
            self.archi_card = SiOptionCardLinear(self)
            self.archi_card.setTitle("archi", "模型架构（已创建，不可修改）")
            self.archi_card.load(safe_get_icon("ic_fluent_branch_fork_filled"))
            self.archi_input = SiLabeledLineEdit(self.archi_card)
            self.archi_input.setTitle("字符串")
            self.archi_input.setText("df-ud")  # 默认值，实际应从模型读取
            self.archi_input.setFixedHeight(48)
            self.archi_input.resize(150, 48)
            self.archi_input.setEnabled(False)  # 设置为不可编辑
            self.archi_card.addWidget(self.archi_input)
            self.archi_card.adjustSize()
            
            # e_dims - 编码器维度大小
            self.e_dims_card = SiOptionCardLinear(self)
            self.e_dims_card.setTitle("e_dims", "编码器维度大小，用卷积提取输入的特征")
            self.e_dims_card.load(safe_get_icon("ic_fluent_box_filled"))
            self.e_dims_input = SiLabeledLineEdit(self.e_dims_card)
            self.e_dims_input.setTitle("整数")
            self.e_dims_input.setText("64")
            self.e_dims_input.setFixedHeight(48)
            self.e_dims_input.resize(150, 48)
            self.e_dims_input.setEnabled(False)  # 设置为不可编辑
            self.e_dims_card.addWidget(self.e_dims_input)
            self.e_dims_card.adjustSize()
            
            # ae_dims - 隐空间维度大小
            self.ae_dims_card = SiOptionCardLinear(self)
            self.ae_dims_card.setTitle("ae_dims", "隐空间维度大小，将编码器卷积拓张为张量\n这个值越大，理论能存的特征越多，但是超过了256大小有用的维度越少")
            self.ae_dims_card.load(safe_get_icon("ic_fluent_cube_filled"))
            self.ae_dims_input = SiLabeledLineEdit(self.ae_dims_card)
            self.ae_dims_input.setTitle("整数")
            self.ae_dims_input.setText("256")
            self.ae_dims_input.setFixedHeight(48)
            self.ae_dims_input.resize(150, 48)
            self.ae_dims_input.setEnabled(False)  # 设置为不可编辑
            self.ae_dims_card.addWidget(self.ae_dims_input)
            self.ae_dims_card.adjustSize()
            
            # d_dims - 解码器维度大小
            self.d_dims_card = SiOptionCardLinear(self)
            self.d_dims_card.setTitle("d_dims", "解码器维度大小，将中间层的输出进行上采样解码\n解码器越大，解码能力越强，模型恢复的图像更厉害")
            self.d_dims_card.load(safe_get_icon("ic_fluent_arrow_upload_filled"))
            self.d_dims_input = SiLabeledLineEdit(self.d_dims_card)
            self.d_dims_input.setTitle("整数")
            self.d_dims_input.setText("128")
            self.d_dims_input.setFixedHeight(48)
            self.d_dims_input.resize(150, 48)
            self.d_dims_input.setEnabled(False)  # 设置为不可编辑
            self.d_dims_card.addWidget(self.d_dims_input)
            self.d_dims_card.adjustSize()
            
            # d_mask_dims - 遮罩解码器维度大小
            self.d_mask_dims_card = SiOptionCardLinear(self)
            self.d_mask_dims_card.setTitle("d_mask_dims", "遮罩解码器维度大小，为pred提供一个遮罩\n过高并不能提高特别多精度")
            self.d_mask_dims_card.load(safe_get_icon("ic_fluent_eye_off_filled"))
            self.d_mask_dims_input = SiLabeledLineEdit(self.d_mask_dims_card)
            self.d_mask_dims_input.setTitle("整数")
            self.d_mask_dims_input.setText("32")
            self.d_mask_dims_input.setFixedHeight(48)
            self.d_mask_dims_input.resize(150, 48)
            self.d_mask_dims_input.setEnabled(False)  # 设置为不可编辑
            self.d_mask_dims_card.addWidget(self.d_mask_dims_input)
            self.d_mask_dims_card.adjustSize()
            
            group.addWidget(self.model_name_card)
            group.addWidget(self.resolution_card)
            group.addWidget(self.archi_card)
            group.addWidget(self.e_dims_card)
            group.addWidget(self.ae_dims_card)
            group.addWidget(self.d_dims_card)
            group.addWidget(self.d_mask_dims_card)
        
        # 数据集参数卡片
        with self.titled_widget_group as group:
            group.addTitle("数据集参数")
            
            # src人脸集数据路径
            self.src_faceset_card = SiOptionCardLinear(self)
            self.src_faceset_card.setTitle("src人脸集数据路径", "源人脸集的路径")
            self.src_faceset_card.load(safe_get_icon("ic_fluent_folder_filled"))
            self.src_faceset_input = SiLabeledLineEdit(self.src_faceset_card)
            self.src_faceset_input.setTitle("路径")
            self.src_faceset_input.setPlaceholderText("请输入src人脸集路径...")
            
            # 设置默认路径为项目根目录下的workspace\data_src\aligned
            project_root = Path(__file__).parent.parent.parent.parent.parent
            default_src_path = project_root / "workspace" / "data_src" / "aligned"
            self.src_faceset_input.textChanged.connect(lambda: self.update_config('src_faceset_path'))
            self.src_faceset_input.setText(str(default_src_path))

            self.src_faceset_input.setFixedHeight(48)
            self.src_faceset_input.resize(500, 48)
            self.src_faceset_card.addWidget(self.src_faceset_input)
            self.src_faceset_card.adjustSize()
            
            # dst人脸集数据路径
            self.dst_faceset_card = SiOptionCardLinear(self)
            self.dst_faceset_card.setTitle("dst人脸集数据路径", "目标人脸集的路径")
            self.dst_faceset_card.load(safe_get_icon("ic_fluent_folder_open_filled"))
            self.dst_faceset_input = SiLabeledLineEdit(self.dst_faceset_card)
            self.dst_faceset_input.setTitle("路径")
            self.dst_faceset_input.setPlaceholderText("请输入dst人脸集路径...")
            
            # 设置默认路径为项目根目录下的workspace\data_dst\aligned
            project_root = Path(__file__).parent.parent.parent.parent.parent
            default_dst_path = project_root / "workspace" / "data_dst" / "aligned"
            self.dst_faceset_input.textChanged.connect(lambda: self.update_config('dst_faceset_path'))
            self.dst_faceset_input.setText(str(default_dst_path))

            self.dst_faceset_input.setFixedHeight(48)
            self.dst_faceset_input.resize(500, 48)
            self.dst_faceset_card.addWidget(self.dst_faceset_input)
            self.dst_faceset_card.adjustSize()
            
            # 人脸类型
            self.face_type_card = SiOptionCardLinear(self)
            self.face_type_card.setTitle("人脸类型", "根据选择的人脸类型对素材进行不同比例的缩放")
            self.face_type_card.load(safe_get_icon("ic_fluent_person_filled"))
            self.face_type_combo = SiCapsuleComboBox(self.face_type_card)
            self.face_type_combo.setTitle("人脸类型")
            self.face_type_combo.setFixedWidth(200)
            self.face_type_combo.setMinimumHeight(32)
            self.face_type_combo.setMaximumHeight(32)
            self.face_type_combo.setEditable(False)
            self.face_type_combo.addItems(["wf", "f", "head", "hf", "ff"])
            self.face_type_combo.setCurrentText("wf")
            self.face_type_combo.currentTextChanged.connect(lambda text: self.update_config('face_type', text))
            self.face_type_card.addWidget(self.face_type_combo)
            self.face_type_card.adjustSize()
            
            group.addWidget(self.src_faceset_card)
            group.addWidget(self.dst_faceset_card)
            group.addWidget(self.face_type_card)
        
        with self.titled_widget_group as group:
            group.addTitle("训练参数")
            
            # 设备（自定义垂直卡片，复选框在标题下方）
            self._device_list = _detect_available_devices()
            self._device_checkboxes = {}

            self.device_card = QWidget(self)
            self.device_card.setStyleSheet(
                f"background-color: {SiGlobal.siui.colors['INTERFACE_BG_C']}; border-radius: 4px;"
            )
            self.device_card.setMinimumWidth(560)
            # 动态计算卡片高度：标题行 56px + 间距 + 复选框 * n
            n_devices = len(self._device_list)
            self.device_card.setMinimumHeight(56 + 16 + n_devices * 36)
            device_card_layout = QVBoxLayout(self.device_card)
            device_card_layout.setContentsMargins(0, 0, 0, 12)
            device_card_layout.setSpacing(0)

            # 标题行（图标 + 主标题 + 副标题）
            title_row = QWidget(self.device_card)
            title_row.setMinimumHeight(56)
            title_row_layout = QHBoxLayout(title_row)
            title_row_layout.setContentsMargins(16, 0, 16, 0)

            icon_label = SiSvgLabel(title_row)
            icon_label.setSvgSize(24, 24)
            icon_label.resize(80, 80)
            ico = safe_get_icon("ic_fluent_desktop_filled")
            if ico:
                icon_label.load(ico)
            title_text = SiLabel(title_row)
            title_text.setStyleSheet(f"color: {SiGlobal.siui.colors['TEXT_A']}; font-size: 14px; font-weight: bold; background: transparent;")
            title_text.setText("设备")
            subtitle_text = SiLabel(title_row)
            subtitle_text.setStyleSheet(f"color: {SiGlobal.siui.colors['TEXT_B']}; font-size: 12px; background: transparent;")
            subtitle_text.setText("勾选要使用的 GPU（可多选），勾选 CPU 则使用 CPU 训练")

            title_row_layout.addWidget(icon_label)
            title_row_layout.addSpacing(8)
            title_row_layout.addWidget(title_text)
            title_row_layout.addSpacing(8)
            title_row_layout.addWidget(subtitle_text)
            title_row_layout.addStretch()

            # 复选框区域
            device_container = QWidget(self.device_card)
            device_container.setMinimumWidth(540)
            device_layout = QVBoxLayout(device_container)
            device_layout.setContentsMargins(28, 0, 0, 0)
            device_layout.setSpacing(4)
            for dev in self._device_list:
                if dev['type'] == 'gpu':
                    label = f"[GPU{dev['index']}] {dev['name']}"
                else:
                    label = f"[CPU] {dev['name']}"
                cb = SiCheckBox(device_container)
                cb.setText(label)
                cb.setMinimumHeight(32)
                cb.setChecked(dev['type'] == 'gpu')
                cb.toggled.connect(lambda checked, d=dev: self._on_device_toggled(d, checked))
                # 强制复选框颜色为紫色，覆盖 siui 默认颜色
                purple = "#9c65ae"
                cb.toggled.connect(lambda checked, c=cb: c.indicator_label.setStyleSheet(
                    f"background-color: {purple}; border-radius: 4px;" if checked
                    else f"border: 1px solid #555555; border-radius: 4px;"
                ))
                if dev['type'] == 'gpu':
                    cb.indicator_label.setStyleSheet(f"background-color: {purple}; border-radius: 4px;")
                    cb.indicator_icon.setVisible(True)
                else:
                    cb.indicator_label.setStyleSheet(f"border: 1px solid #555555; border-radius: 4px;")
                    cb.indicator_icon.setVisible(False)
                device_layout.addWidget(cb)
                self._device_checkboxes[f"{dev['type']}_{dev['index']}"] = cb

            device_card_layout.addWidget(title_row)
            device_card_layout.addSpacing(4)
            device_card_layout.addWidget(device_container)
            device_card_layout.addStretch()

            # 批次大小
            self.batch_size_card = SiOptionCardLinear(self)
            self.batch_size_card.setTitle("批次大小", "每次训练的样本数量")
            self.batch_size_card.load(safe_get_icon("ic_fluent_table_simple_filled"))
            self.batch_size_input = SiLabeledLineEdit(self.batch_size_card)
            self.batch_size_input.setTitle("整数")
            self.batch_size_input.setText("8")
            self.batch_size_input.setFixedHeight(48)
            self.batch_size_input.resize(150, 48)
            self.batch_size_input.textChanged.connect(lambda: self.update_config('batch_size'))
            self.batch_size_card.addWidget(self.batch_size_input)
            self.batch_size_card.adjustSize()
            
            # 目标迭代数
            self.target_iter_card = SiOptionCardLinear(self)
            self.target_iter_card.setTitle("目标迭代数", "设为0表示无限训练")
            self.target_iter_card.load(safe_get_icon("ic_fluent_target_filled"))
            self.target_iter_input = SiLabeledLineEdit(self.target_iter_card)
            self.target_iter_input.setTitle("整数")
            self.target_iter_input.setText("0")
            self.target_iter_input.setFixedHeight(48)
            self.target_iter_input.resize(150, 48)
            self.target_iter_input.textChanged.connect(lambda: self.update_config('target_iter'))
            self.target_iter_card.addWidget(self.target_iter_input)
            self.target_iter_card.adjustSize()
            
            # 自动保存间隔
            self.auto_save_card = SiOptionCardLinear(self)
            self.auto_save_card.setTitle("自动保存间隔", "每N次迭代自动保存模型")
            self.auto_save_card.load(safe_get_icon("ic_fluent_save_filled"))
            self.auto_save_input = SiLabeledLineEdit(self.auto_save_card)
            self.auto_save_input.setTitle("整数")
            self.auto_save_input.setText("1000")
            self.auto_save_input.setFixedHeight(48)
            self.auto_save_input.resize(150, 48)
            self.auto_save_input.textChanged.connect(lambda: self.update_config('auto_save_interval'))
            self.auto_save_card.addWidget(self.auto_save_input)
            self.auto_save_card.adjustSize()

            # 训练精度选择
            self.precision_card = SiOptionCardLinear(self)
            self.precision_card.setTitle("训练精度", "选择训练精度（fp32/bf16）")
            self.precision_card.load(safe_get_icon("ic_fluent_number_row_filled"))
            self.precision_combo = SiCapsuleComboBox(self.precision_card)
            self.precision_combo.setTitle("训练精度")
            self.precision_combo.setFixedWidth(200)
            self.precision_combo.setMinimumHeight(32)
            self.precision_combo.setMaximumHeight(32)
            self.precision_combo.setEditable(False)
            self.precision_combo.addItems(["fp32", "bf16"])
            self.precision_combo.setCurrentText("fp32")
            self.precision_combo.currentTextChanged.connect(lambda text: self.update_config('precision', text))
            self.precision_card.addWidget(self.precision_combo)
            self.precision_card.adjustSize()
            
            # 梯度检查点开关
            self.gradient_checkpointing_card = SiOptionCardLinear(self)
            self.gradient_checkpointing_card.setTitle("梯度检查点", "启用梯度检查点以节省显存")
            self.gradient_checkpointing_card.load(safe_get_icon("ic_fluent_memory_filled"))
            self.gradient_checkpointing_switch = SiSwitch(self.gradient_checkpointing_card)
            self.gradient_checkpointing_switch.setChecked(False)
            self.gradient_checkpointing_switch.toggled.connect(lambda state: self.update_config('gradient_checkpointing'))
            self.gradient_checkpointing_card.addWidget(self.gradient_checkpointing_switch)
            self.gradient_checkpointing_card.adjustSize()
            
            # 梯度裁剪开关
            self.clipgrad_card = SiOptionCardLinear(self)
            self.clipgrad_card.setTitle("梯度裁剪", "防止梯度爆炸")
            self.clipgrad_card.load(safe_get_icon("ic_fluent_clipboard_task_list_ltr_filled"))
            self.clipgrad_switch = SiSwitch(self.clipgrad_card)
            self.clipgrad_switch.setChecked(True)
            self.clipgrad_switch.toggled.connect(lambda state: self.update_config('clipgrad'))
            self.clipgrad_card.addWidget(self.clipgrad_switch)
            self.clipgrad_card.adjustSize()
            
            # 嘴眼优先开关
            self.prioritize_mouth_eyes_card = SiOptionCardLinear(self)
            self.prioritize_mouth_eyes_card.setTitle("嘴眼优先", "优先训练嘴部和眼部区域")
            self.prioritize_mouth_eyes_card.load(safe_get_icon("ic_fluent_eye_filled"))
            self.prioritize_mouth_eyes_switch = SiSwitch(self.prioritize_mouth_eyes_card)
            self.prioritize_mouth_eyes_switch.setChecked(True)
            self.prioritize_mouth_eyes_switch.toggled.connect(lambda state: self.update_config('prioritize_mouth_eyes'))
            self.prioritize_mouth_eyes_card.addWidget(self.prioritize_mouth_eyes_switch)
            self.prioritize_mouth_eyes_card.adjustSize()

            # === 遮罩训练开关 ===
            self.masked_training_card = SiOptionCardLinear(self)
            self.masked_training_card.setTitle("遮罩训练", "启用后仅在遮罩区域内计算损失")
            self.masked_training_card.load(safe_get_icon("ic_fluent_shield_filled"))
            self.masked_training_switch = SiSwitch(self.masked_training_card)
            self.masked_training_switch.setChecked(True)
            self.masked_training_switch.toggled.connect(lambda state: self.update_config('masked_training'))
            self.masked_training_card.addWidget(self.masked_training_switch)
            self.masked_training_card.adjustSize()

            # === 遮罩羽化开关 ===
            self.blur_out_mask_card = SiOptionCardLinear(self)
            self.blur_out_mask_card.setTitle("遮罩羽化", "对输入的mask外周进行高斯模糊")
            self.blur_out_mask_card.load(safe_get_icon("ic_fluent_blur_filled"))
            self.blur_out_mask_switch = SiSwitch(self.blur_out_mask_card)
            self.blur_out_mask_switch.setChecked(False)
            self.blur_out_mask_switch.toggled.connect(lambda state: self.update_config('blur_out_mask'))
            self.blur_out_mask_card.addWidget(self.blur_out_mask_switch)
            self.blur_out_mask_card.adjustSize()

            # === GPU优化器开关 ===
            self.models_opt_on_gpu_card = SiOptionCardLinear(self)
            self.models_opt_on_gpu_card.setTitle("优化器驻留GPU", "将优化器状态放在GPU显存中以加速训练")
            self.models_opt_on_gpu_card.load(safe_get_icon("ic_fluent_developer_board_filled"))
            self.models_opt_on_gpu_switch = SiSwitch(self.models_opt_on_gpu_card)
            self.models_opt_on_gpu_switch.setChecked(True)
            self.models_opt_on_gpu_switch.toggled.connect(lambda state: self.update_config('models_opt_on_gpu'))
            self.models_opt_on_gpu_card.addWidget(self.models_opt_on_gpu_switch)
            self.models_opt_on_gpu_card.adjustSize()

            # === 快速生成器开关 ===
            self.use_fast_generator_card = SiOptionCardLinear(self)
            self.use_fast_generator_card.setTitle("快速生成器", "使用优化后的快速数据生成器")
            self.use_fast_generator_card.load(safe_get_icon("ic_fluent_flash_filled"))
            self.use_fast_generator_switch = SiSwitch(self.use_fast_generator_card)
            self.use_fast_generator_switch.setChecked(False)
            self.use_fast_generator_switch.toggled.connect(lambda state: self.update_config('use_fast_generator'))
            self.use_fast_generator_card.addWidget(self.use_fast_generator_switch)
            self.use_fast_generator_card.adjustSize()

            # === 预览历史开关 ===
            self.write_preview_history_card = SiOptionCardLinear(self)
            self.write_preview_history_card.setTitle("写入预览历史", "将预览图保存到历史文件夹")
            self.write_preview_history_card.load(safe_get_icon("ic_fluent_history_filled"))
            self.write_preview_history_switch = SiSwitch(self.write_preview_history_card)
            self.write_preview_history_switch.setChecked(False)
            self.write_preview_history_switch.toggled.connect(lambda state: self.update_config('write_preview_history'))
            self.write_preview_history_card.addWidget(self.write_preview_history_switch)
            self.write_preview_history_card.adjustSize()

            # === 预训练模式 ===
            self.pretrain_card = SiOptionCardLinear(self)
            self.pretrain_card.setTitle("预训练模式", "使用通用人脸数据进行预训练")
            self.pretrain_card.load(safe_get_icon("ic_fluent_hat_graduation_filled"))
            self.pretrain_switch = SiSwitch(self.pretrain_card)
            self.pretrain_switch.setChecked(False)
            self.pretrain_switch.toggled.connect(lambda state: self.update_config('pretrain'))
            self.pretrain_card.addWidget(self.pretrain_switch)
            self.pretrain_card.adjustSize()

            # === 崩溃阈值 ===
            self.crash_threshold_card = SiOptionCardLinear(self)
            self.crash_threshold_card.setTitle("崩溃阈值", "当loss大于阈值时，视为训练崩溃，此时不会自动保存覆盖备份，防止备份丢失")
            self.crash_threshold_card.load(safe_get_icon("ic_fluent_warning_filled"))
            self.crash_threshold_input = SiLabeledLineEdit(self.crash_threshold_card)
            self.crash_threshold_input.setTitle("小数")
            self.crash_threshold_input.setText("0.0")
            self.crash_threshold_input.setFixedHeight(48)
            self.crash_threshold_input.resize(150, 48)
            self.crash_threshold_input.textChanged.connect(lambda: self.update_config('crash_threshold'))
            self.crash_threshold_card.addWidget(self.crash_threshold_input)
            self.crash_threshold_card.adjustSize()

            # === 最大备份数 ===
            self.max_backups_card = SiOptionCardLinear(self)
            self.max_backups_card.setTitle("最大备份数", "最多保留的自动备份数量")
            self.max_backups_card.load(safe_get_icon("ic_fluent_copy_filled"))
            self.max_backups_input = SiLabeledLineEdit(self.max_backups_card)
            self.max_backups_input.setTitle("整数")
            self.max_backups_input.setText("3")
            self.max_backups_input.setFixedHeight(48)
            self.max_backups_input.resize(150, 48)
            self.max_backups_input.textChanged.connect(lambda: self.update_config('max_backups'))
            self.max_backups_card.addWidget(self.max_backups_input)
            self.max_backups_card.adjustSize()

            group.addWidget(self.device_card)
            group.addWidget(self.batch_size_card)
            group.addWidget(self.precision_card)
            group.addWidget(self.gradient_checkpointing_card)
            group.addWidget(self.clipgrad_card)
            group.addWidget(self.prioritize_mouth_eyes_card)
            group.addWidget(self.target_iter_card)
            group.addWidget(self.auto_save_card)
            group.addWidget(self.masked_training_card)
            group.addWidget(self.blur_out_mask_card)
            group.addWidget(self.models_opt_on_gpu_card)
            group.addWidget(self.use_fast_generator_card)
            group.addWidget(self.write_preview_history_card)
            group.addWidget(self.pretrain_card)
            group.addWidget(self.crash_threshold_card)
            group.addWidget(self.max_backups_card)

        with self.titled_widget_group as group:
            group.addTitle("优化器参数")

            # AdaBelief优化器开关
            self.adabelief_card = SiOptionCardLinear(self)
            self.adabelief_card.setTitle("AdaBelief优化器", "启用AdaBelief优化算法")
            self.adabelief_card.load(safe_get_icon("ic_fluent_toggle_left_filled"))
            self.adabelief_switch = SiSwitch(self.adabelief_card)
            self.adabelief_switch.setChecked(False)
            self.adabelief_switch.toggled.connect(lambda state: self.update_config('use_adabelief'))
            self.adabelief_card.addWidget(self.adabelief_switch)
            self.adabelief_card.adjustSize()
            
            # 学习率
            self.learning_rate_card = SiOptionCardLinear(self)
            self.learning_rate_card.setTitle("学习率", "控制参数更新的步长")
            self.learning_rate_card.load(safe_get_icon("ic_fluent_trending_up_filled"))
            self.learning_rate_input = SiLabeledLineEdit(self.learning_rate_card)
            self.learning_rate_input.setTitle("小数")
            self.learning_rate_input.setText("0.00005")
            self.learning_rate_input.setFixedHeight(48)
            self.learning_rate_input.resize(150, 48)
            self.learning_rate_input.textChanged.connect(lambda: self.update_config('learning_rate'))
            self.learning_rate_card.addWidget(self.learning_rate_input)
            self.learning_rate_card.adjustSize()
            
            # 余弦退火周期
            self.lr_cos_card = SiOptionCardLinear(self)
            self.lr_cos_card.setTitle("余弦退火周期", "余弦退火周期迭代步数（0=关闭）")
            self.lr_cos_card.load(safe_get_icon("ic_fluent_calendar_schedule_filled"))
            self.lr_cos_input = SiLabeledLineEdit(self.lr_cos_card)
            self.lr_cos_input.setTitle("整数")
            self.lr_cos_input.setText("0")
            self.lr_cos_input.setFixedHeight(48)
            self.lr_cos_input.resize(150, 48)
            self.lr_cos_input.textChanged.connect(lambda: self.update_config('lr_cos'))
            self.lr_cos_card.addWidget(self.lr_cos_input)
            self.lr_cos_card.adjustSize()

            # === 学习率丢弃策略 ===
            self.lr_dropout_card = SiOptionCardLinear(self)
            self.lr_dropout_card.setTitle("学习率丢弃", "LR dropout 策略（y/n/cpu）")
            self.lr_dropout_card.load(safe_get_icon("ic_fluent_trending_down_filled"))
            self.lr_dropout_combo = SiCapsuleComboBox(self.lr_dropout_card)
            self.lr_dropout_combo.setTitle("策略")
            self.lr_dropout_combo.setFixedWidth(200)
            self.lr_dropout_combo.setMinimumHeight(32)
            self.lr_dropout_combo.setMaximumHeight(32)
            self.lr_dropout_combo.setEditable(False)
            self.lr_dropout_combo.addItems(["n", "y", "cpu"])
            self.lr_dropout_combo.setCurrentText("n")
            self.lr_dropout_combo.currentTextChanged.connect(lambda text: self.update_config('lr_dropout', text))
            self.lr_dropout_card.addWidget(self.lr_dropout_combo)
            self.lr_dropout_card.adjustSize()

            group.addWidget(self.adabelief_card)
            group.addWidget(self.learning_rate_card)
            group.addWidget(self.lr_cos_card)
            group.addWidget(self.lr_dropout_card)

        with self.titled_widget_group as group:
            group.addTitle("GAN训练参数")

            # GAN强度
            self.gan_power_card = SiOptionCardLinear(self)
            self.gan_power_card.setTitle("GAN强度", "控制GAN损失的权重")
            self.gan_power_card.load(safe_get_icon("ic_fluent_gauge_filled"))
            self.gan_power_input = SiLabeledLineEdit(self.gan_power_card)
            self.gan_power_input.setTitle("小数")
            self.gan_power_input.setText("0.004")
            self.gan_power_input.setFixedHeight(48)
            self.gan_power_input.resize(150, 48)
            self.gan_power_input.textChanged.connect(lambda: self.update_config('gan_power'))
            self.gan_power_card.addWidget(self.gan_power_input)
            self.gan_power_card.adjustSize()

            # === GAN 块大小 ===
            self.gan_patch_size_card = SiOptionCardLinear(self)
            self.gan_patch_size_card.setTitle("GAN块大小", "GAN 鉴别器的 patch 大小")
            self.gan_patch_size_card.load(safe_get_icon("ic_fluent_grid_filled"))
            self.gan_patch_size_input = SiLabeledLineEdit(self.gan_patch_size_card)
            self.gan_patch_size_input.setTitle("整数")
            self.gan_patch_size_input.setText("32")
            self.gan_patch_size_input.setFixedHeight(48)
            self.gan_patch_size_input.resize(150, 48)
            self.gan_patch_size_input.textChanged.connect(lambda: self.update_config('gan_patch_size'))
            self.gan_patch_size_card.addWidget(self.gan_patch_size_input)
            self.gan_patch_size_card.adjustSize()

            # === GAN 维度 ===
            self.gan_dims_card = SiOptionCardLinear(self)
            self.gan_dims_card.setTitle("GAN维度", "GAN 鉴别器的基础通道数")
            self.gan_dims_card.load(safe_get_icon("ic_fluent_cube_filled"))
            self.gan_dims_input = SiLabeledLineEdit(self.gan_dims_card)
            self.gan_dims_input.setTitle("整数")
            self.gan_dims_input.setText("16")
            self.gan_dims_input.setFixedHeight(48)
            self.gan_dims_input.resize(150, 48)
            self.gan_dims_input.textChanged.connect(lambda: self.update_config('gan_dims'))
            self.gan_dims_card.addWidget(self.gan_dims_input)
            self.gan_dims_card.adjustSize()

            group.addWidget(self.gan_power_card)
            group.addWidget(self.gan_patch_size_card)
            group.addWidget(self.gan_dims_card)

        with self.titled_widget_group as group:
            group.addTitle("补丁参数")

            # === 真脸强度 ===
            self.true_face_power_card = SiOptionCardLinear(self)
            self.true_face_power_card.setTitle("真脸强度", "CodeDiscriminator 真脸损失权重（0=关闭）")
            self.true_face_power_card.load(safe_get_icon("ic_fluent_emoji_filled"))
            self.true_face_power_input = SiLabeledLineEdit(self.true_face_power_card)
            self.true_face_power_input.setTitle("小数")
            self.true_face_power_input.setText("0.0")
            self.true_face_power_input.setFixedHeight(48)
            self.true_face_power_input.resize(150, 48)
            self.true_face_power_input.textChanged.connect(lambda: self.update_config('true_face_power'))
            self.true_face_power_card.addWidget(self.true_face_power_input)
            self.true_face_power_card.adjustSize()

            # === 面部风格强度 ===
            self.face_style_power_card = SiOptionCardLinear(self)
            self.face_style_power_card.setTitle("面部风格强度", "风格迁移中面部风格的权重（0=关闭）")
            self.face_style_power_card.load(safe_get_icon("ic_fluent_style_filled"))
            self.face_style_power_input = SiLabeledLineEdit(self.face_style_power_card)
            self.face_style_power_input.setTitle("小数")
            self.face_style_power_input.setText("0.0")
            self.face_style_power_input.setFixedHeight(48)
            self.face_style_power_input.resize(150, 48)
            self.face_style_power_input.textChanged.connect(lambda: self.update_config('face_style_power'))
            self.face_style_power_card.addWidget(self.face_style_power_input)
            self.face_style_power_card.adjustSize()

            # === 背景风格强度 ===
            self.bg_style_power_card = SiOptionCardLinear(self)
            self.bg_style_power_card.setTitle("背景风格强度", "风格迁移中背景风格的权重（0=关闭）")
            self.bg_style_power_card.load(safe_get_icon("ic_fluent_image_filled"))
            self.bg_style_power_input = SiLabeledLineEdit(self.bg_style_power_card)
            self.bg_style_power_input.setTitle("小数")
            self.bg_style_power_input.setText("0.0")
            self.bg_style_power_input.setFixedHeight(48)
            self.bg_style_power_input.resize(150, 48)
            self.bg_style_power_input.textChanged.connect(lambda: self.update_config('bg_style_power'))
            self.bg_style_power_card.addWidget(self.bg_style_power_input)
            self.bg_style_power_card.adjustSize()

            group.addWidget(self.true_face_power_card)
            group.addWidget(self.face_style_power_card)
            group.addWidget(self.bg_style_power_card)

        with self.titled_widget_group as group:
            group.addTitle("数据增强参数")

            # 旋转范围
            self.rotation_range_card = SiOptionCardLinear(self)
            self.rotation_range_card.setTitle("旋转范围", "随机旋转角度范围（度）")
            self.rotation_range_card.load(safe_get_icon("ic_fluent_arrow_rotate_clockwise_filled"))
            self.rotation_range_input = SiLabeledLineEdit(self.rotation_range_card)
            self.rotation_range_input.setTitle("整数")
            self.rotation_range_input.setText("20")
            self.rotation_range_input.setFixedHeight(48)
            self.rotation_range_input.resize(150, 48)
            self.rotation_range_input.textChanged.connect(lambda: self.update_config('rotation_range'))
            self.rotation_range_card.addWidget(self.rotation_range_input)
            self.rotation_range_card.adjustSize()

            # 缩放范围
            self.scale_range_card = SiOptionCardLinear(self)
            self.scale_range_card.setTitle("缩放范围", "随机缩放比例范围")
            self.scale_range_card.load(safe_get_icon("ic_fluent_resize_filled"))
            self.scale_range_input = SiLabeledLineEdit(self.scale_range_card)
            self.scale_range_input.setTitle("小数")
            self.scale_range_input.setText("0.15")
            self.scale_range_input.setFixedHeight(48)
            self.scale_range_input.resize(150, 48)
            self.scale_range_input.textChanged.connect(lambda: self.update_config('scale_range'))
            self.scale_range_card.addWidget(self.scale_range_input)
            self.scale_range_card.adjustSize()

            # 偏移范围
            self.t_range_card = SiOptionCardLinear(self)
            self.t_range_card.setTitle("偏移范围", "随机平移偏移范围")
            self.t_range_card.load(safe_get_icon("ic_fluent_arrow_move_filled"))
            self.t_range_input = SiLabeledLineEdit(self.t_range_card)
            self.t_range_input.setTitle("小数")
            self.t_range_input.setText("0.05")
            self.t_range_input.setFixedHeight(48)
            self.t_range_input.resize(150, 48)
            self.t_range_input.textChanged.connect(lambda: self.update_config('t_range'))
            self.t_range_card.addWidget(self.t_range_input)
            self.t_range_card.adjustSize()

            # 颜色模式
            self.ct_mode_card = SiOptionCardLinear(self)
            self.ct_mode_card.setTitle("颜色模式", "色彩转换模式")
            self.ct_mode_card.load(safe_get_icon("ic_fluent_color_filled"))
            self.ct_mode_combo = SiCapsuleComboBox(self.ct_mode_card)
            self.ct_mode_combo.setTitle("颜色模式")
            self.ct_mode_combo.setFixedWidth(200)
            self.ct_mode_combo.setMinimumHeight(32)
            self.ct_mode_combo.setMaximumHeight(32)
            self.ct_mode_combo.setEditable(False)
            self.ct_mode_combo.addItems(["none", "rct", "lct", "mkl", "idt", "sot"])
            self.ct_mode_combo.setCurrentText("rct")
            self.ct_mode_combo.currentTextChanged.connect(lambda text: self.update_config('ct_mode', text))
            self.ct_mode_card.addWidget(self.ct_mode_combo)
            self.ct_mode_card.adjustSize()

            # 随机HSV增强
            self.random_hsv_card = SiOptionCardLinear(self)
            self.random_hsv_card.setTitle("随机HSV增强", "启用随机色相/饱和度/亮度调整")
            self.random_hsv_card.load(safe_get_icon("ic_fluent_weather_sunny_filled"))
            self.random_hsv_switch = SiSwitch(self.random_hsv_card)
            self.random_hsv_switch.setChecked(False)
            self.random_hsv_switch.toggled.connect(lambda state: self.update_config('random_hsv'))
            self.random_hsv_card.addWidget(self.random_hsv_switch)
            self.random_hsv_card.adjustSize()

            group.addWidget(self.rotation_range_card)
            group.addWidget(self.scale_range_card)
            group.addWidget(self.t_range_card)
            group.addWidget(self.ct_mode_card)
            group.addWidget(self.random_hsv_card)

            # 随机扭曲
            self.random_warp_card = SiOptionCardLinear(self)
            self.random_warp_card.setTitle("随机扭曲", "启用随机变形增强")
            self.random_warp_card.load(safe_get_icon("ic_fluent_arrow_swap_filled"))
            self.random_warp_switch = SiSwitch(self.random_warp_card)
            self.random_warp_switch.setChecked(True)
            self.random_warp_switch.toggled.connect(lambda state: self.update_config('random_warp'))
            self.random_warp_card.addWidget(self.random_warp_switch)
            self.random_warp_card.adjustSize()

            # 随机src翻转
            self.random_src_flip_card = SiOptionCardLinear(self)
            self.random_src_flip_card.setTitle("随机src翻转", "源图像随机水平翻转")
            self.random_src_flip_card.load(safe_get_icon("ic_fluent_arrow_sync_filled"))
            self.random_src_flip_switch = SiSwitch(self.random_src_flip_card)
            self.random_src_flip_switch.setChecked(False)
            self.random_src_flip_switch.toggled.connect(lambda state: self.update_config('random_src_flip'))
            self.random_src_flip_card.addWidget(self.random_src_flip_switch)
            self.random_src_flip_card.adjustSize()

            # 随机dst翻转
            self.random_dst_flip_card = SiOptionCardLinear(self)
            self.random_dst_flip_card.setTitle("随机dst翻转", "目标图像随机水平翻转")
            self.random_dst_flip_card.load(safe_get_icon("ic_fluent_arrow_split_filled"))
            self.random_dst_flip_switch = SiSwitch(self.random_dst_flip_card)
            self.random_dst_flip_switch.setChecked(False)
            self.random_dst_flip_switch.toggled.connect(lambda state: self.update_config('random_dst_flip'))
            self.random_dst_flip_card.addWidget(self.random_dst_flip_switch)
            self.random_dst_flip_card.adjustSize()

            # 平均偏航角分布
            self.uniform_yaw_card = SiOptionCardLinear(self)
            self.uniform_yaw_card.setTitle("平均偏航角分布", "统一人脸偏航角分布")
            self.uniform_yaw_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.uniform_yaw_switch = SiSwitch(self.uniform_yaw_card)
            self.uniform_yaw_switch.setChecked(False)
            self.uniform_yaw_switch.toggled.connect(lambda state: self.update_config('uniform_yaw_distribution'))
            self.uniform_yaw_card.addWidget(self.uniform_yaw_switch)
            self.uniform_yaw_card.adjustSize()

            group.addWidget(self.random_warp_card)
            group.addWidget(self.random_src_flip_card)
            group.addWidget(self.random_dst_flip_card)
            group.addWidget(self.uniform_yaw_card)

        # ======== 冻结层参数 ========
        with self.titled_widget_group as group:
            group.addTitle("冻结层参数")

            # 冻结编码器 (所有架构通用)
            self.freeze_encoder_card = SiOptionCardLinear(self)
            self.freeze_encoder_card.setTitle("冻结编码器 (Encoder)", "冻结编码器层，节省显存/算力，为解码器提供稳定特征")
            self.freeze_encoder_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_encoder_switch = SiSwitch(self.freeze_encoder_card)
            self.freeze_encoder_switch.setChecked(False)
            self.freeze_encoder_switch.toggled.connect(lambda state: self.update_config('freeze_encoder'))
            self.freeze_encoder_card.addWidget(self.freeze_encoder_switch)
            self.freeze_encoder_card.adjustSize()

            # 冻结中间层 (SAEHD)
            self.freeze_inter_card = SiOptionCardLinear(self)
            self.freeze_inter_card.setTitle("冻结中间层 (Inter)", "冻结中间层：df 架构为 Inter，liae 架构为 Inter_AB + Inter_B")
            self.freeze_inter_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_inter_switch = SiSwitch(self.freeze_inter_card)
            self.freeze_inter_switch.setChecked(False)
            self.freeze_inter_switch.toggled.connect(lambda state: self.update_config('freeze_inter'))
            self.freeze_inter_card.addWidget(self.freeze_inter_switch)
            self.freeze_inter_card.adjustSize()

            # 冻结中间层AB (LIAE)
            self.freeze_inter_AB_card = SiOptionCardLinear(self)
            self.freeze_inter_AB_card.setTitle("冻结中间层AB (Inter AB)", "冻结 LIAE 的 Inter_AB 层")
            self.freeze_inter_AB_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_inter_AB_switch = SiSwitch(self.freeze_inter_AB_card)
            self.freeze_inter_AB_switch.setChecked(False)
            self.freeze_inter_AB_switch.toggled.connect(lambda state: self.update_config('freeze_inter_AB'))
            self.freeze_inter_AB_card.addWidget(self.freeze_inter_AB_switch)
            self.freeze_inter_AB_card.adjustSize()

            # 冻结中间层B (LIAE)
            self.freeze_inter_B_card = SiOptionCardLinear(self)
            self.freeze_inter_B_card.setTitle("冻结中间层B (Inter B)", "冻结 LIAE 的 Inter_B 层")
            self.freeze_inter_B_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_inter_B_switch = SiSwitch(self.freeze_inter_B_card)
            self.freeze_inter_B_switch.setChecked(False)
            self.freeze_inter_B_switch.toggled.connect(lambda state: self.update_config('freeze_inter_B'))
            self.freeze_inter_B_card.addWidget(self.freeze_inter_B_switch)
            self.freeze_inter_B_card.adjustSize()

            # 冻结源中间层 (AMP)
            self.freeze_inter_src_card = SiOptionCardLinear(self)
            self.freeze_inter_src_card.setTitle("冻结源中间层 (Inter Src)", "冻结 AMP 的源中间层")
            self.freeze_inter_src_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_inter_src_switch = SiSwitch(self.freeze_inter_src_card)
            self.freeze_inter_src_switch.setChecked(False)
            self.freeze_inter_src_switch.toggled.connect(lambda state: self.update_config('freeze_inter_src'))
            self.freeze_inter_src_card.addWidget(self.freeze_inter_src_switch)
            self.freeze_inter_src_card.adjustSize()

            # 冻结目标中间层 (AMP)
            self.freeze_inter_dst_card = SiOptionCardLinear(self)
            self.freeze_inter_dst_card.setTitle("冻结目标中间层 (Inter Dst)", "冻结 AMP 的目标中间层")
            self.freeze_inter_dst_card.load(safe_get_icon("ic_fluent_people_team_filled"))
            self.freeze_inter_dst_switch = SiSwitch(self.freeze_inter_dst_card)
            self.freeze_inter_dst_switch.setChecked(False)
            self.freeze_inter_dst_switch.toggled.connect(lambda state: self.update_config('freeze_inter_dst'))
            self.freeze_inter_dst_card.addWidget(self.freeze_inter_dst_switch)
            self.freeze_inter_dst_card.adjustSize()

            # 冻结解码器Mask分支 (所有架构通用)
            self.freeze_decoder_mask_card = SiOptionCardLinear(self)
            self.freeze_decoder_mask_card.setTitle("冻结解码器Mask分支 (Decoder Mask)", "冻结解码器的遮罩分支参数，仅影响mask预测通道")
            self.freeze_decoder_mask_card.load(safe_get_icon("ic_fluent_eye_off_filled"))
            self.freeze_decoder_mask_switch = SiSwitch(self.freeze_decoder_mask_card)
            self.freeze_decoder_mask_switch.setChecked(False)
            self.freeze_decoder_mask_switch.toggled.connect(lambda state: self.update_config('freeze_decoder_mask'))
            self.freeze_decoder_mask_card.addWidget(self.freeze_decoder_mask_switch)
            self.freeze_decoder_mask_card.adjustSize()

            # 根据架构显示/隐藏不同冻结选项
            _fc_model_class = self.model_info.get('class_name', 'SAEHD')
            _fc_is_liae = _fc_model_class == 'SAEHD' and 'liae' in self.config_data.get('archi', '')
            _fc_is_df = _fc_model_class == 'SAEHD' and not _fc_is_liae
            self.freeze_inter_card.setVisible(_fc_is_df)
            self.freeze_inter_AB_card.setVisible(_fc_is_liae)
            self.freeze_inter_B_card.setVisible(_fc_is_liae)
            self.freeze_inter_src_card.setVisible(_fc_model_class == 'AMP')
            self.freeze_inter_dst_card.setVisible(_fc_model_class == 'AMP')

            group.addWidget(self.freeze_encoder_card)
            if _fc_is_df:
                group.addWidget(self.freeze_inter_card)
            if _fc_is_liae:
                group.addWidget(self.freeze_inter_AB_card)
                group.addWidget(self.freeze_inter_B_card)
            if _fc_model_class == 'AMP':
                group.addWidget(self.freeze_inter_src_card)
                group.addWidget(self.freeze_inter_dst_card)
            group.addWidget(self.freeze_decoder_mask_card)

        self.content().setAttachment(self.titled_widget_group)
        
        # 控制面板
        self.start_button = SiPushButton(self)
        self.start_button.resize(128, 32)
        self.start_button.attachment().setText("开始训练")
        self.start_button.clicked.connect(self.on_start_training)
        
        self.cancel_button = SiPushButton(self)
        self.cancel_button.resize(128, 32)
        self.cancel_button.attachment().setText("取消")
        self.cancel_button.clicked.connect(self.closeParentLayer)
        
        self.panel().addWidget(self.cancel_button, "left")
        self.panel().addWidget(self.start_button, "right")
        
        # 从模型 dat 文件读取的配置覆盖控件默认值
        self._populate_from_config()

        # 加载样式表
        SiGlobal.siui.reloadStyleSheetRecursively(self)

    def on_start_training(self):
        """构建 CLI 命令并在新控制台窗口中启动训练"""
        try:
            self.update_config('device')  # 确保设备状态最新
            cfg = self.config_data

            # 检查路径是否为空
            src_path = cfg.get('src_faceset_path', '')
            dst_path = cfg.get('dst_faceset_path', '')
            if not src_path or not dst_path:
                print("[ERROR] src 和 dst 人脸集路径不能为空")
                return

            python_exe = sys.executable
            model_class = self.model_info.get('class_name', 'SAEHD')
            project_root = Path(__file__).parent.parent.parent.parent.parent

            args = [
                python_exe, "main.py", "train",
                "--model", model_class,
                "--training-data-src-dir", src_path,
                "--training-data-dst-dir", dst_path,
                "--model-dir", self._model_dir,
                "--force-model-name", self._model_name,
                "--no-preview",
                "--silent-start",
                "--webui-port", self._webui_port,
                "--webui-password", self._webui_password,
            ]

            device_val = cfg.get('device', '0')
            if device_val == 'cpu':
                args.append("--cpu-only")
            elif device_val:
                args.extend(["--force-gpu-idxs", device_val])

            # Write GUI-editable options to model's _data.dat before starting
            _fc_model_class = self.model_info.get('class_name', 'SAEHD')
            _fc_dat_path = Path(self._model_dir) / f"{self._model_name}_{_fc_model_class}_data.dat"
            if _fc_dat_path.exists():
                try:
                    import pickle
                    _fc_data = pickle.loads(_fc_dat_path.read_bytes())
                    if 'options' in _fc_data:
                        # Freeze options (bool)
                        _fc_data['options']['freeze_encoder'] = self.freeze_encoder_switch.isChecked()
                        _fc_data['options']['freeze_decoder_mask'] = self.freeze_decoder_mask_switch.isChecked()
                        if _fc_model_class == 'SAEHD':
                            if 'liae' in str(self.config_data.get('archi', '')):
                                _fc_data['options']['freeze_inter_AB'] = self.freeze_inter_AB_switch.isChecked()
                                _fc_data['options']['freeze_inter_B'] = self.freeze_inter_B_switch.isChecked()
                            else:
                                _fc_data['options']['freeze_inter'] = self.freeze_inter_switch.isChecked()
                        else:  # AMP
                            _fc_data['options']['freeze_inter_src'] = self.freeze_inter_src_switch.isChecked()
                            _fc_data['options']['freeze_inter_dst'] = self.freeze_inter_dst_switch.isChecked()
                        # Runtime options from config_data
                        # Key mapping: (config_data_key, model_option_key, type_cast)
                        _runtime_opts = [
                            ('crash_threshold',           'crash_threshold',     float),
                            ('auto_save_interval',        'backup_interval',     int),
                            ('max_backups',               'max_backups',         int),
                            ('target_iter',               'target_iter',         int),
                            ('batch_size',                'batch_size',          int),
                            ('learning_rate',             'learning_rate',       float),
                            ('gan_power',                 'gan_power',           float),
                            ('prioritize_mouth_eyes',     'eyes_mouth_prio',     bool),
                            ('uniform_yaw_distribution',  'uniform_yaw',         bool),
                            ('masked_training',           'masked_training',     bool),
                            ('blur_out_mask',             'blur_out_mask',       bool),
                            ('random_warp',               'random_warp',         bool),
                            ('true_face_power',           'true_face_power',     float),
                            ('face_style_power',          'face_style_power',    float),
                            ('bg_style_power',            'bg_style_power',      float),
                            ('models_opt_on_gpu',         'models_opt_on_gpu',   bool),
                            ('use_fast_generator',        'use_fast_generator',  bool),
                            ('write_preview_history',     'write_preview_history', bool),
                            ('lr_dropout',                'lr_dropout',          str),
                            ('gan_patch_size',            'gan_patch_size',      int),
                            ('gan_dims',                  'gan_dims',            int),
                        ]
                        for _ckey, _mkey, _cast in _runtime_opts:
                            if _ckey in self.config_data:
                                _raw = self.config_data[_ckey]
                                try:
                                    _fc_data['options'][_mkey] = _cast(_raw)
                                except (ValueError, TypeError):
                                    pass  # skip invalid value
                        _fc_tmp = _fc_dat_path.with_suffix('.dat.write_tmp')
                        _fc_tmp.write_bytes(pickle.dumps(_fc_data))
                        _fc_tmp.replace(_fc_dat_path)
                except Exception as _fc_e:
                    print(f"[ERROR] 写入配置到 _data.dat 失败: {_fc_e}")

            self.closeParentLayer()

            cmd_line = subprocess.list2cmdline(args) + " & pause"
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                subprocess.Popen(['cmd', '/c', cmd_line], creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=str(project_root))
            else:
                subprocess.Popen(['cmd', '/c', cmd_line], cwd=str(project_root))

        except Exception as e:
            print(f"[ERROR] 启动训练失败: {e}")

    def _on_device_toggled(self, dev, checked):
        """设备复选框互斥：GPU 与 CPU 不能同时选；至少保留一个设备被选中"""
        # 如果正在取消选中，检查是否至少还有一个设备被选中
        if not checked:
            any_other_checked = any(
                cb.isChecked()
                for key, cb in self._device_checkboxes.items()
                if key != f"{dev['type']}_{dev['index']}"
            )
            if not any_other_checked:
                # 这是最后一个选中的设备，阻止取消
                self._device_checkboxes[f"{dev['type']}_{dev['index']}"].setChecked(True)
                return

        if dev['type'] == 'cpu' and checked:
            for key, cb in self._device_checkboxes.items():
                if key.startswith('gpu_'):
                    cb.setChecked(False)
        elif dev['type'] == 'gpu' and checked:
            cpu_key = 'cpu_'
            if cpu_key in self._device_checkboxes:
                self._device_checkboxes[cpu_key].setChecked(False)
        self.update_config('device')

    def _get_device_value(self):
        """从复选框状态生成设备配置字符串"""
        selected_gpus = []
        cpu_selected = False
        for dev in self._device_list:
            key = "{}_{}".format(dev['type'], dev['index'])
            cb = self._device_checkboxes.get(key)
            if cb and cb.isChecked():
                if dev['type'] == 'gpu':
                    selected_gpus.append(str(dev['index']))
                elif dev['type'] == 'cpu':
                    cpu_selected = True
        if cpu_selected:
            return 'cpu'
        if selected_gpus:
            return ','.join(selected_gpus)
        return 'cpu'

    def _restore_device_config(self, device_val):
        """从配置字符串恢复复选框状态"""
        device_str = str(device_val)
        if device_str == 'cpu':
            cpu_key = 'cpu_'
            if cpu_key in self._device_checkboxes:
                self._device_checkboxes[cpu_key].setChecked(True)
        else:
            gpu_indices = device_str.split(',')
            for key, cb in self._device_checkboxes.items():
                if key.startswith('gpu_'):
                    idx = key.split('_', 1)[1]
                    cb.setChecked(idx in gpu_indices)


    @staticmethod
    def _opt(info, key, default):
        """从模型 info 获取值，'?' 或 None 时视为未找到，返回 default"""
        v = info.get(key)
        if v is None or v == '?':
            return default
        return v

    def _build_config_data(self):
        """从模型信息构建配置数据，缺失的项用默认值填充"""
        info = self.model_info
        _ = self._opt  # shorthand

        # 训练精度：use_bf16 → precision
        use_bf16 = _(info, 'use_bf16', False)
        if not isinstance(use_bf16, bool):
            use_bf16 = False
        precision = 'bf16' if use_bf16 else 'fp32'

        # 学习率可能以 numpy float 存储，统一转成字符串
        lr_raw = _(info, 'lr', 5e-5)
        try:
            lr_str = f'{float(lr_raw):.8f}'.rstrip('0').rstrip('.')
        except (TypeError, ValueError):
            lr_str = '0.00005'

        # GAN 强度
        gan_raw = _(info, 'gan_power', 0.004)
        try:
            gan_float = float(gan_raw)
            gan_power_str = str(gan_float)
        except (TypeError, ValueError):
            gan_power_str = '0.004'

        # 随机 HSV：random_hsv_power > 0 时启用
        hsv_raw = _(info, 'random_hsv_power', 0)
        try:
            hsv_enabled = float(hsv_raw) > 0
        except (TypeError, ValueError):
            hsv_enabled = False

        return {
            'model_name': _(info, 'name', self._model_name),
            'resolution': str(_(info, 'resolution', '256')),
            'archi': str(_(info, 'archi', 'df-ud')),
            'e_dims': str(_(info, 'e_dims', '64')),
            'ae_dims': str(_(info, 'ae_dims', '256')),
            'd_dims': str(_(info, 'd_dims', '128')),
            'd_mask_dims': str(_(info, 'd_mask_dims', '32')),
            'device': '0',
            'batch_size': str(_(info, 'batch_size', '8')),
            'precision': precision,
            'gradient_checkpointing': bool(_(info, 'gradient_checkpointing', False)),
            'clipgrad': bool(_(info, 'clipgrad', True)),
            'target_iter': str(_(info, 'target_iter', '0')),
            'auto_save_interval': str(_(info, 'backup_interval', '1000')),
            'learning_rate': lr_str,
            'lr_cos': str(_(info, 'lr_cos', '0')),
            'ct_mode': str(_(info, 'ct_mode', 'rct')),
            'random_hsv': hsv_enabled,
            'use_adabelief': bool(_(info, 'adabelief', False)),
            'gan_power': gan_power_str,
            'random_warp': bool(_(info, 'random_warp', True)),
            'random_src_flip': bool(_(info, 'random_src_flip', False)),
            'random_dst_flip': bool(_(info, 'random_dst_flip', False)),
            'uniform_yaw_distribution': bool(_(info, 'uniform_yaw', False)),
            'rotation_range': '20',
            'scale_range': '0.15',
            't_range': '0.05',
            'prioritize_mouth_eyes': bool(_(info, 'eyes_mouth_prio', True)),
            # 缺失的模型参数
            'masked_training': bool(_(info, 'masked_training', True)),
            'blur_out_mask': bool(_(info, 'blur_out_mask', False)),
            'models_opt_on_gpu': bool(_(info, 'models_opt_on_gpu', True)),
            'use_fast_generator': bool(_(info, 'use_fast_generator', False)),
            'write_preview_history': bool(_(info, 'write_preview_history', False)),
            'pretrain': bool(_(info, 'pretrain', False)),
            'lr_dropout': str(_(info, 'lr_dropout', 'n')),
            'true_face_power': str(_(info, 'true_face_power', '0.0')),
            'face_style_power': str(_(info, 'face_style_power', '0.0')),
            'bg_style_power': str(_(info, 'bg_style_power', '0.0')),
            'crash_threshold': str(_(info, 'crash_threshold', '0.0')),
            'max_backups': str(_(info, 'max_backups', '3')),
            'gan_patch_size': str(_(info, 'gan_patch_size', '32')),
            'gan_dims': str(_(info, 'gan_dims', '16')),
            'src_faceset_path': '',
            'dst_faceset_path': '',
            'face_type': str(_(info, 'face_type', 'wf')),
            # 冻结层参数（默认不冻结）
            'freeze_encoder': bool(_(info, 'freeze_encoder', False)),
            'freeze_decoder_mask': bool(_(info, 'freeze_decoder_mask', False)),
            'freeze_inter': bool(_(info, 'freeze_inter', False)),
            'freeze_inter_AB': bool(_(info, 'freeze_inter_AB', False)),
            'freeze_inter_B': bool(_(info, 'freeze_inter_B', False)),
            'freeze_inter_src': bool(_(info, 'freeze_inter_src', False)),
            'freeze_inter_dst': bool(_(info, 'freeze_inter_dst', False)),
        }

    def _populate_from_config(self):
        """用 config_data 中的值覆盖所有控件（替换硬编码默认值）"""
        cfg = self.config_data

        # 文本输入框
        self.resolution_input.setText(str(cfg.get('resolution', '256')))
        self.archi_input.setText(str(cfg.get('archi', 'df-ud')))
        self.e_dims_input.setText(str(cfg.get('e_dims', '64')))
        self.ae_dims_input.setText(str(cfg.get('ae_dims', '256')))
        self.d_dims_input.setText(str(cfg.get('d_dims', '128')))
        self.d_mask_dims_input.setText(str(cfg.get('d_mask_dims', '32')))
        self.batch_size_input.setText(str(cfg.get('batch_size', '8')))
        self.target_iter_input.setText(str(cfg.get('target_iter', '0')))
        self.auto_save_input.setText(str(cfg.get('auto_save_interval', '1000')))
        self.learning_rate_input.setText(str(cfg.get('learning_rate', '0.00005')))
        self.gan_power_input.setText(str(cfg.get('gan_power', '0.004')))
        self._restore_device_config(cfg.get('device', '0'))
        self.rotation_range_input.setText(str(cfg.get('rotation_range', '20')))
        self.scale_range_input.setText(str(cfg.get('scale_range', '0.15')))
        self.t_range_input.setText(str(cfg.get('t_range', '0.05')))

        # 下拉框
        self.face_type_combo.setCurrentText(str(cfg.get('face_type', 'wf')))
        self.precision_combo.setCurrentText(str(cfg.get('precision', 'fp32')))
        self.ct_mode_combo.setCurrentText(str(cfg.get('ct_mode', 'rct')))

        # 开关
        self.gradient_checkpointing_switch.setChecked(bool(cfg.get('gradient_checkpointing', False)))
        self.clipgrad_switch.setChecked(bool(cfg.get('clipgrad', True)))
        self.prioritize_mouth_eyes_switch.setChecked(bool(cfg.get('prioritize_mouth_eyes', True)))
        self.adabelief_switch.setChecked(bool(cfg.get('use_adabelief', False)))
        self.lr_cos_input.setText(str(cfg.get('lr_cos', '0')))
        self.random_hsv_switch.setChecked(bool(cfg.get('random_hsv', False)))
        self.random_warp_switch.setChecked(bool(cfg.get('random_warp', True)))
        self.random_src_flip_switch.setChecked(bool(cfg.get('random_src_flip', False)))
        self.random_dst_flip_switch.setChecked(bool(cfg.get('random_dst_flip', False)))
        self.uniform_yaw_switch.setChecked(bool(cfg.get('uniform_yaw_distribution', False)))

        # 新增训练参数开关
        self.masked_training_switch.setChecked(bool(cfg.get('masked_training', True)))
        self.blur_out_mask_switch.setChecked(bool(cfg.get('blur_out_mask', False)))
        self.models_opt_on_gpu_switch.setChecked(bool(cfg.get('models_opt_on_gpu', True)))
        self.use_fast_generator_switch.setChecked(bool(cfg.get('use_fast_generator', False)))
        self.write_preview_history_switch.setChecked(bool(cfg.get('write_preview_history', False)))
        self.pretrain_switch.setChecked(bool(cfg.get('pretrain', False)))

        # 文本输入框（新）
        self.crash_threshold_input.setText(str(cfg.get('crash_threshold', '0.0')))
        self.max_backups_input.setText(str(cfg.get('max_backups', '3')))
        self.true_face_power_input.setText(str(cfg.get('true_face_power', '0.0')))
        self.face_style_power_input.setText(str(cfg.get('face_style_power', '0.0')))
        self.bg_style_power_input.setText(str(cfg.get('bg_style_power', '0.0')))
        self.gan_patch_size_input.setText(str(cfg.get('gan_patch_size', '32')))
        self.gan_dims_input.setText(str(cfg.get('gan_dims', '16')))

        # 下拉框（新）
        self.lr_dropout_combo.setCurrentText(str(cfg.get('lr_dropout', 'n')))

        # 冻结层开关
        self.freeze_encoder_switch.setChecked(bool(cfg.get("freeze_encoder", False)))
        self.freeze_decoder_mask_switch.setChecked(bool(cfg.get("freeze_decoder_mask", False)))
        self.freeze_inter_switch.setChecked(bool(cfg.get("freeze_inter", False)))
        self.freeze_inter_AB_switch.setChecked(bool(cfg.get("freeze_inter_AB", False)))
        self.freeze_inter_B_switch.setChecked(bool(cfg.get("freeze_inter_B", False)))
        self.freeze_inter_src_switch.setChecked(bool(cfg.get("freeze_inter_src", False)))
        self.freeze_inter_dst_switch.setChecked(bool(cfg.get("freeze_inter_dst", False)))

    def get_config(self):
        """获取当前配置"""
        return self.config_data.copy()
    
    def update_config(self, key, value=None):
        """更新配置数据"""
        try:
            # 如果提供了value，直接使用；否则从控件中获取
            if value is not None:
                self.config_data[key] = value
            else:
                # 根据键名找到对应的控件并更新配置
                if key == 'model_name':
                    self.config_data[key] = self.model_name_input.text()
                elif key == 'resolution':
                    self.config_data[key] = self.resolution_input.text()
                elif key == 'src_faceset_path':
                    self.config_data[key] = self.src_faceset_input.text()
                elif key == 'dst_faceset_path':
                    self.config_data[key] = self.dst_faceset_input.text()
                elif key == 'face_type':
                    self.config_data[key] = self.face_type_combo.currentText()
                elif key == 'device':
                    self.config_data[key] = self._get_device_value()
                elif key == 'batch_size':
                    self.config_data[key] = self.batch_size_input.text()
                elif key == 'target_iter':
                    self.config_data[key] = self.target_iter_input.text()
                elif key == 'auto_save_interval':
                    self.config_data[key] = self.auto_save_input.text()
                elif key == 'learning_rate':
                    self.config_data[key] = self.learning_rate_input.text()
                elif key == 'rotation_range':
                    self.config_data[key] = self.rotation_range_input.text()
                elif key == 'scale_range':
                    self.config_data[key] = self.scale_range_input.text()
                elif key == 't_range':
                    self.config_data[key] = self.t_range_input.text()
                elif key == 'gan_power':
                    self.config_data[key] = self.gan_power_input.text()
                elif key == 'gradient_checkpointing':
                    self.config_data[key] = self.gradient_checkpointing_switch.isChecked()
                elif key == 'clipgrad':
                    self.config_data[key] = self.clipgrad_switch.isChecked()
                elif key == 'lr_cos':
                    self.config_data[key] = self.lr_cos_input.text()
                elif key == 'random_hsv':
                    self.config_data[key] = self.random_hsv_switch.isChecked()
                elif key == 'use_adabelief':
                    self.config_data[key] = self.adabelief_switch.isChecked()
                elif key == 'random_warp':
                    self.config_data[key] = self.random_warp_switch.isChecked()
                elif key == 'random_src_flip':
                    self.config_data[key] = self.random_src_flip_switch.isChecked()
                elif key == 'random_dst_flip':
                    self.config_data[key] = self.random_dst_flip_switch.isChecked()
                elif key == 'uniform_yaw_distribution':
                    self.config_data[key] = self.uniform_yaw_switch.isChecked()
                elif key == 'prioritize_mouth_eyes':
                    self.config_data[key] = self.prioritize_mouth_eyes_switch.isChecked()
                elif key == 'ct_mode':
                    self.config_data[key] = self.ct_mode_combo.currentText()
                elif key == 'masked_training':
                    self.config_data[key] = self.masked_training_switch.isChecked()
                elif key == 'blur_out_mask':
                    self.config_data[key] = self.blur_out_mask_switch.isChecked()
                elif key == 'models_opt_on_gpu':
                    self.config_data[key] = self.models_opt_on_gpu_switch.isChecked()
                elif key == 'use_fast_generator':
                    self.config_data[key] = self.use_fast_generator_switch.isChecked()
                elif key == 'write_preview_history':
                    self.config_data[key] = self.write_preview_history_switch.isChecked()
                elif key == 'pretrain':
                    self.config_data[key] = self.pretrain_switch.isChecked()
                elif key == 'crash_threshold':
                    self.config_data[key] = self.crash_threshold_input.text()
                elif key == 'max_backups':
                    self.config_data[key] = self.max_backups_input.text()
                elif key == 'lr_dropout':
                    self.config_data[key] = self.lr_dropout_combo.currentText()
                elif key == 'true_face_power':
                    self.config_data[key] = self.true_face_power_input.text()
                elif key == 'face_style_power':
                    self.config_data[key] = self.face_style_power_input.text()
                elif key == 'bg_style_power':
                    self.config_data[key] = self.bg_style_power_input.text()
                elif key == 'gan_patch_size':
                    self.config_data[key] = self.gan_patch_size_input.text()
                elif key == 'gan_dims':
                    self.config_data[key] = self.gan_dims_input.text()
                elif key == 'freeze_encoder':
                    self.config_data[key] = self.freeze_encoder_switch.isChecked()
                elif key == 'freeze_inter':
                    self.config_data[key] = self.freeze_inter_switch.isChecked()
                elif key == 'freeze_inter_AB':
                    self.config_data[key] = self.freeze_inter_AB_switch.isChecked()
                elif key == 'freeze_inter_B':
                    self.config_data[key] = self.freeze_inter_B_switch.isChecked()
                elif key == 'freeze_inter_src':
                    self.config_data[key] = self.freeze_inter_src_switch.isChecked()
                elif key == 'freeze_inter_dst':
                    self.config_data[key] = self.freeze_inter_dst_switch.isChecked()
                elif key == 'freeze_decoder_mask':
                    self.config_data[key] = self.freeze_decoder_mask_switch.isChecked()

        except Exception as e:
            print(f"[ERROR] 更新配置失败: {key}, 错误: {e}")


