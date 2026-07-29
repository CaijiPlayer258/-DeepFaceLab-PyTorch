"""
新建模型配置子页面 - 点击新建模型按钮时弹出（仅模型架构）
"""
from PyQt5.QtCore import pyqtSignal
from siui.components.page.child_page import SiChildPage
from siui.components import SiTitledWidgetGroup, SiPushButton
from siui.components.editbox import SiLabeledLineEdit
from siui.components.option_card import SiOptionCardLinear
from siui.components.combobox_ import SiCapsuleComboBox
from siui.core import SiGlobal


def safe_get_icon(name):
    """安全获取图标"""
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


class NewModelConfigChildPage(SiChildPage):
    """新建模型配置子页面 - 点击新建模型按钮时弹出（仅模型架构）"""
    
    # 定义信号：当用户点击创建模型按钮时发出
    model_created = pyqtSignal(dict)
    
    def __init__(self, parent=None, model_name="model_001", model_info=None):
        super().__init__(parent)
        self.model_info = model_info or {}
        
        self.view().setMinimumWidth(900)
        self.content().setTitle(f"新建模型配置 - {model_name}")
        self.content().setPadding(48)
        
        # 存储所有配置项的值
        self.config_data = {
            'model_name': model_name,
            'resolution': '256',
            'archi': 'DF',
            'subarchi': '-ud',
            'e_dims': '64',
            'ae_dims': '256',
            'd_dims': '128',
            'd_mask_dims': '32'
        }
        
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
            
            # archi - 模型架构（胶囊下拉框）
            self.archi_card = SiOptionCardLinear(self)
            self.archi_card.setTitle("架构", "选择你的模型架构")
            self.archi_card.load(safe_get_icon("ic_fluent_branch_fork_filled"))
            self.archi_combo = SiCapsuleComboBox(self.archi_card)
            self.archi_combo.setTitle("架构")
            self.archi_combo.setFixedWidth(200)
            self.archi_combo.setMinimumHeight(32)
            self.archi_combo.setMaximumHeight(32)
            self.archi_combo.setEditable(False)
            self.archi_combo.addItems(["DF", "LIAE", "AMP"])
            self.archi_combo.setCurrentText("DF")
            self.archi_combo.currentTextChanged.connect(lambda text: self.update_architecture(text))
            self.archi_card.addWidget(self.archi_combo)
            self.archi_card.adjustSize()
            
            # 子分支选项（胶囊下拉框）
            self.subarchi_card = SiOptionCardLinear(self)
            self.subarchi_card.setTitle("子分支", "选择架构的子分支变体")
            self.subarchi_card.load(safe_get_icon("ic_fluent_branch_compare_filled"))
            self.subarchi_combo = SiCapsuleComboBox(self.subarchi_card)
            self.subarchi_combo.setTitle("子分支")
            self.subarchi_combo.setFixedWidth(200)
            self.subarchi_combo.setMinimumHeight(32)
            self.subarchi_combo.setMaximumHeight(32)
            self.subarchi_combo.setEditable(False)
            self.subarchi_combo.addItems(["-u", "-ud", "-ut", "-udt", "-d", "-dt", "-t"])
            self.subarchi_combo.setCurrentText("-ud")
            self.subarchi_combo.currentTextChanged.connect(lambda text: self.update_subarchitecture(text))
            self.subarchi_card.addWidget(self.subarchi_combo)
            self.subarchi_card.adjustSize()
            
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
            group.addWidget(self.subarchi_card)
            group.addWidget(self.e_dims_card)
            group.addWidget(self.ae_dims_card)
            group.addWidget(self.d_dims_card)
            group.addWidget(self.d_mask_dims_card)
        
        self.content().setAttachment(self.titled_widget_group)
        
        # 控制面板
        self.create_button = SiPushButton(self)
        self.create_button.resize(128, 32)
        self.create_button.attachment().setText("创建模型")
        self.create_button.clicked.connect(self.on_create_clicked)
        
        self.cancel_button = SiPushButton(self)
        self.cancel_button.resize(128, 32)
        self.cancel_button.attachment().setText("取消")
        self.cancel_button.clicked.connect(self.closeParentLayer)
        
        self.panel().addWidget(self.cancel_button, "left")
        self.panel().addWidget(self.create_button, "right")
        
        # 加载样式表
        SiGlobal.siui.reloadStyleSheetRecursively(self)
    
    def update_architecture(self, text):
        """更新架构选择并改变描述"""
        self.config_data['archi'] = text
        print(f"[CONFIG] archi = {text}")
        
        # 根据选择的架构更新副标题描述
        descriptions = {
            "DF": "编码器-inter-双解码器架构",
            "LIAE": "编码器-双inter-解码器架构",
            "AMP": "编码器-双inter(身份分离)-解码器架构"
        }
        if text in descriptions:
            self.archi_card.setTitle("架构", descriptions[text])
    
    def update_subarchitecture(self, text):
        """更新子分支选择并改变描述"""
        self.config_data['subarchi'] = text
        print(f"[CONFIG] subarchi = {text}")
        
        # 根据选择的子分支更新副标题描述
        descriptions = {
            "-u": "像素进行归一化处理",
            "-d": "提供一种可学习的上采样",
            "-t": "编码器增加一次下采样",
            "-ud": "像素归一化 + 可学习上采样",
            "-ut": "像素归一化 + 编码器增加下采样",
            "-udt": "像素归一化 + 可学习上采样 + 编码器增加下采样 ",
            "-dt": "可学习上采样 + 编码器增加@下采样"
        }
        if text in descriptions:
            self.subarchi_card.setTitle("子分支", descriptions[text])
    
    def update_config(self, key, value=None):
        """更新配置值"""
        # 检查是否是开关
        if hasattr(self, f'{key}_switch'):
            switch_widget = getattr(self, f'{key}_switch')
            self.config_data[key] = switch_widget.isChecked()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
        # 检查是否是输入框
        elif hasattr(self, f'{key}_input'):
            input_widget = getattr(self, f'{key}_input')
            self.config_data[key] = input_widget.text()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
        # 检查是否是下拉框（SiCapsuleComboBox）
        elif hasattr(self, f'{key}_combo'):
            combo_widget = getattr(self, f'{key}_combo')
            if value is not None:
                self.config_data[key] = value
            else:
                # 对于 SiCapsuleComboBox，使用 currentText() 获取当前值
                self.config_data[key] = combo_widget.currentText()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
    
    def get_config(self):
        """获取当前配置"""
        return self.config_data.copy()
    
    def on_create_clicked(self):
        """点击创建模型按钮"""
        
        # 发出信号，传递配置数据
        self.model_created.emit(self.config_data)
        
        # 关闭子页面
        self.closeParentLayer()
        
