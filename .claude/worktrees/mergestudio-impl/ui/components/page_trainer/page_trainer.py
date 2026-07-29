"""
DeepFaceLab Torch - Trainer Page
训练器页面：包含FaceModel和XSegModel两个子页面
"""

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QByteArray, QVariantAnimation
from PyQt5.QtWidgets import QBoxLayout, QLabel, QStackedWidget, QWidget, QVBoxLayout, QSizePolicy, QApplication, QGraphicsOpacityEffect, QComboBox
from pathlib import Path
from contextlib import contextmanager
from siui.components.page import SiPage
from siui.components.page.child_page import SiChildPage
from siui.components.container import SiTriSectionPanelCard, SiDenseContainer
from siui.components.editbox import SiLabeledLineEdit
from siui.components.button import SiPushButtonRefactor
from siui.components import SiTitledWidgetGroup, SiPushButton, SiSwitch
from siui.components.combobox import SiComboBox
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.option_card import SiOptionCardLinear, SiOptionCardPlane
from siui.components.widgets.navigation_bar import SiNavigationBarH
from siui.core import SiGlobal, Si, SiExpAnimation, SiColor

# 导入独立的子页面类
from .components.training_config_page import TrainingConfigChildPage
from .components.new_model_config_page import NewModelConfigChildPage


@contextmanager
def createPanelCard(parent: SiTitledWidgetGroup, title: str) -> SiTriSectionPanelCard:
    """创建面板卡片的上下文管理器"""
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)


def safe_get_icon(name):
    """安全获取图标"""
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


class TrainerPage(SiPage):
    """
    Trainer主页面
    包含水平导航栏，可在FaceModel和XSegModel之间切换
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("训练器")
        
        # 创建导航栏（在滚动区域外）
        self.nav_bar = SiNavigationBarH(self)
        self.nav_bar.addItem("FaceModel")
        self.nav_bar.addItem("XSegModel")
        self.nav_bar.setCurrentIndex(0)
        self.nav_bar.indexChanged.connect(self._on_nav_changed)
        
        # 设置导航栏初始位置
        title_height = 80
        padding = self.padding
        nav_bar_y = title_height  # 增加间距，避免被标题挡住
        self.nav_bar.adjustSize()
        nav_height = self.nav_bar.height()
        self.nav_bar.setGeometry(padding, nav_bar_y, 800, nav_height)  # 初始宽度800
        
        # 提升导航栏的层级，确保它在最前面
        self.nav_bar.raise_()
        
        # 给导航栏添加背景
        from siui.core import SiColor, SiGlobal
        self.nav_bar.setStyleSheet(f"""
            background-color: {SiGlobal.siui.colors["INTERFACE_BG_B"]}
        """)
        
        # 创建滚动容器
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)
        
        # 初始化卡片高度动画字典
        self.card_height_animations = {}
        
        # 模型计数器（用于生成新模型名称）
        self.model_counter = 4
        self._models_info = {}  # 模型名称 → 模型信息字典的查找表
        
        # 模型配置分组
        with self.titled_widgets_group as group:
            group.addTitle("模型配置")
            
            # FaceModel内容 - 数据路径卡片
            with createPanelCard(group, "数据路径") as card:
                container = SiDenseContainer(card.body())
                container.layout().setDirection(QBoxLayout.TopToBottom)
                container.layout().setSpacing(12)
                container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                
                # 模型目录输入框
                model_dir_input = SiLabeledLineEdit(card)
                model_dir_input.setTitle("模型目录")
                model_dir_input.setPlaceholderText("请输入模型目录路径...")

                # 使用相对于项目根目录的默认路径
                project_root = Path(__file__).parent.parent.parent.parent
                default_model_dir = project_root / "workspace" / "model"
                model_dir_input.setText(str(default_model_dir))
                model_dir_input.resize(700, 64)
                self._model_dir_input = model_dir_input
                self._model_dir = str(default_model_dir)

                # 目录路径变化时重新扫描模型列表
                model_dir_input.textChanged.connect(self._on_model_dir_changed)
                
                container.addWidget(model_dir_input)
                card.body().addWidget(container)
            
            
            # WebUI 设置卡片
            with createPanelCard(group, "WebUI 设置") as card:
                container = SiDenseContainer(card.body())
                container.layout().setDirection(QBoxLayout.TopToBottom)
                container.layout().setSpacing(12)
                container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                
                # WebUI 端口输入框
                webui_port_input = SiLabeledLineEdit(card)
                webui_port_input.setTitle("WebUI 端口")
                webui_port_input.setPlaceholderText("默认: 6789")
                webui_port_input.setText("6789")
                webui_port_input.resize(700, 64)
                self._webui_port_input = webui_port_input
                self._webui_port = "6789"
                webui_port_input.textChanged.connect(lambda t: setattr(self, '_webui_port', t or '6789'))
                container.addWidget(webui_port_input)
                
                # WebUI 密码输入框
                webui_pwd_input = SiLabeledLineEdit(card)
                webui_pwd_input.setTitle("WebUI 密码")
                webui_pwd_input.setPlaceholderText("默认: caiji")
                webui_pwd_input.setText("caiji")
                webui_pwd_input.resize(700, 64)
                self._webui_pwd_input = webui_pwd_input
                self._webui_password = "caiji"
                webui_pwd_input.textChanged.connect(lambda t: setattr(self, '_webui_password', t or 'caiji'))
                container.addWidget(webui_pwd_input)
                
                card.body().addWidget(container)
            
            # Models列表卡片
            with createPanelCard(group, "Models") as card:
                # 设置初始最小宽度为1000（与页面默认一致）
                card.setMinimumWidth(1000)
                
                models_container = SiDenseContainer(card.body())
                models_container.layout().setDirection(QBoxLayout.TopToBottom)
                models_container.layout().setSpacing(16)
                models_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
                
                # 保存引用以便后续添加新模型
                self.models_container = models_container
                self.models_card = card
                
                # 记录卡片当前高度，用于定时器检测变化
                self.last_models_card_height = card.height()

                # 扫描模型目录中的已保存模型实例
                self._rebuild_model_list(str(default_model_dir))

                # 在底部添加"新建模型"按钮
                new_model_button = SiPushButtonRefactor(card)
                new_model_button.setText("新建模型")
                new_model_button.setSvgIcon(safe_get_icon("ic_fluent_cube_add_filled"))
                new_model_button.adjustSize()
                new_model_button.clicked.connect(self.on_new_model_clicked)
                models_container.addWidget(new_model_button)
                self._new_model_btn_face = new_model_button
                
                card.body().addWidget(models_container)
        
        # 设置最小尺寸避免除以零错误
        self.titled_widgets_group.setMinimumSize(800, 600)
        
        # 创建定时器，定期检查Models卡片高度变化
        self.height_check_timer = QTimer(self)
        self.height_check_timer.setInterval(100)  # 每100毫秒检查一次
        self.height_check_timer.timeout.connect(self.check_models_card_height)
        self.height_check_timer.start()
        
        # 使用setAttachment
        self.setAttachment(self.titled_widgets_group)
    
    def _fade_out_cards_then(self, callback):
        """淡出所有现有卡片，完成后执行回调"""
        layout = self.models_container.layout()
        if layout is None:
            callback()
            return
        cards = []
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w and isinstance(w, (SiOptionCardLinear, SiPushButtonRefactor, QLabel)):
                cards.append(w)
        if not cards:
            callback()
            return

        done = [0]
        total = len(cards)
        for card in cards:
            anim = QPropertyAnimation(card, b"windowOpacity")
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setDuration(150)
            def on_finished(c=card, a=anim):
                layout.removeWidget(c)
                c.deleteLater()
                done[0] += 1
                if done[0] >= total:
                    callback()
            anim.finished.connect(on_finished)
            anim.start()

    def _on_nav_changed(self, index: int):
        """导航栏切换 FaceModel / XSegModel"""
        # 删除旧卡片和按钮（保留容器的内部 stretch/spacer）
        layout = self.models_container.layout()
        if layout:
            for i in range(layout.count() - 1, -1, -1):
                w = layout.itemAt(i).widget()
                if w and isinstance(w, (SiOptionCardLinear, SiPushButtonRefactor, QLabel)):
                    layout.removeWidget(w)
                    w.hide()
                    w.deleteLater()
        QApplication.processEvents()
        if index == 0:
            self._rebuild_model_list(self._model_dir)
            # Re-add FaceModel button (was deleted above)
            btn = SiPushButtonRefactor(self.models_card)
            btn.setText("新建模型")
            btn.setSvgIcon(safe_get_icon("ic_fluent_cube_add_filled"))
            btn.adjustSize()
            btn.clicked.connect(self.on_new_model_clicked)
            self.models_container.addWidget(btn)
            self._new_model_btn_face = btn
        elif index == 1:
            self._rebuild_xseg_model_list(self._model_dir)
        self.models_card.adjustSize()
        self.models_container.adjustSize()
        self.models_container.updateGeometry()
        if hasattr(self, 'titled_widgets_group'):
            self.titled_widgets_group.adjustSize()
            self.titled_widgets_group.updateGeometry()
        self._animate_models_card_height()

    def _rebuild_xseg_model_list(self, model_dir: str):
        """扫描并重建 XSeg/XSegLite 模型列表"""
        from pathlib import Path
        from datetime import datetime
        import os, glob as _glob

        layout = self.models_container.layout()
        if layout is None:
            return
        # 移除旧卡片和占位符
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            widget = item.widget() if item else None
            if widget and (isinstance(widget, SiOptionCardLinear) or isinstance(widget, SiPushButtonRefactor) or isinstance(widget, QLabel)):
                layout.removeWidget(widget)
                widget.deleteLater()

        base = Path(model_dir)
        xseg_models = []
        seen = set()

        # Scan for XSeg models — one card per model (skip _opt and auxiliary .pth files)
        for subdir_name in ['XSegLite', 'XSeg', 'GAXSeg']:
            subdir = base / subdir_name
            if not subdir.exists():
                continue
            for pth_file in subdir.glob('*.pth'):
                if '_opt' in pth_file.name or '_star' in pth_file.name or '_4stage' in pth_file.name:
                    continue
                stem = pth_file.stem
                # Normalize: "XSegLite_256" → dedup key
                parts = stem.rsplit('_', 1)
                key = parts[0] if len(parts) == 2 else stem
                if key in seen:
                    continue
                seen.add(key)
                stat = pth_file.stat()
                xseg_models.append({
                    'name': stem,
                    'type': subdir_name.upper(),
                    'class_name': subdir_name.upper(),
                    'resolution': '256',
                    'archi': 'Conv3x3 4-stage',
                    'face_type': 'wf',
                    'iter': 0,
                    'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size': stat.st_size,
                })

        if not xseg_models:
            # Show placeholder
            placeholder = QLabel("未检测到 XSeg/XSegLite 模型\n请在 workspace/model/XSegLite/ 下放置 .pth 文件")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 32px;")
            self.models_container.addWidget(placeholder)
        else:
            for info in xseg_models:
                card = self._create_xseg_model_card(info)
                self.models_container.addWidget(card)

        # Replace old "新建" button if exists (use try to handle deleted C++ object)
        try:
            if hasattr(self, '_new_xseg_btn') and self._new_xseg_btn is not None:
                self._new_xseg_btn.hide()
                self._new_xseg_btn.deleteLater()
        except RuntimeError:
            pass
        self._new_xseg_btn = None
        new_btn = SiPushButtonRefactor(self.models_card)
        new_btn.setText("新建 XSeg 模型")
        new_btn.setSvgIcon(safe_get_icon("ic_fluent_cube_add_filled"))
        new_btn.adjustSize()
        new_btn.clicked.connect(lambda: self._start_xseg_training())
        self.models_container.addWidget(new_btn)
        self._new_xseg_btn = new_btn

        self._animate_models_card_height()

    def _create_xseg_model_card(self, info: dict) -> SiOptionCardLinear:
        card = SiOptionCardLinear(self.models_card)
        title = f"{info['name']}  ({info['type']})"
        subtitle = f"架构: {info['archi']}  |  res: {info['resolution']}  |  {info['size']/1048576:.1f}MB"
        card.setTitle(title, "分辨率: 任意 | " + subtitle.replace("res: 256 | ", ""))
        card.load(safe_get_icon("ic_fluent_layer_filled"))

        train_btn = SiPushButtonRefactor(card)
        train_btn.setText("训练")
        train_btn.setSvgIcon(safe_get_icon("ic_fluent_caret_right_regular"))
        train_btn.adjustSize()
        train_btn.clicked.connect(lambda: self._on_xseg_model_clicked(info))
        card.addWidget(train_btn)
        card.adjustSize()
        return card

    def _on_xseg_model_clicked(self, model_info: dict):
        """点击 XSeg 模型卡片 → 打开配置页"""
        from .components.training_config_page import TrainingConfigChildPage
        model_name = model_info.get('name', 'XSegLite')
        main_window = SiGlobal.siui.windows.get("MAIN_WINDOW")
        if main_window and hasattr(main_window, 'layerChildPage'):
            child_page = TrainingConfigChildPage(self, model_name, model_info=model_info,
                                                  model_dir=self._model_dir,
                                                  webui_port=self._webui_port,
                                                  webui_password=self._webui_password)
            main_window.layerChildPage().setChildPage(child_page)

    def _start_xseg_training(self):
        """直接启动 XSegLite 训练（CLI 模式）"""
        import subprocess, sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        cmd = [
            sys.executable, str(project_root / "main.py"), "train",
            "--model", "XSegLite",
            "--model-dir", self._model_dir,
            "--training-data-src-dir", str(project_root / "workspace" / "data_src" / "aligned"),
            "--training-data-dst-dir", str(project_root / "workspace" / "data_src" / "aligned"),
            "--no-preview", "--silent-start",
        ]
        if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=str(project_root))
        else:
            subprocess.Popen(cmd, cwd=str(project_root))

    def resizeEvent(self, event):
        """调整导航栏宽度和滚动区域"""
        super().resizeEvent(event)
        
        # 获取页面尺寸
        page_width = event.size().width()
        page_height = event.size().height()
        padding = self.padding
        
        # 只调整导航栏的宽度，保持Y位置和高度不变
        current_geo = self.nav_bar.geometry()
        nav_width = page_width - 2 * padding
        self.nav_bar.setGeometry(padding, current_geo.y(), nav_width, current_geo.height())
        
        # 确保导航栏在最前面
        self.nav_bar.raise_()
        
        # 强制刷新导航栏布局
        self.nav_bar.update()
        self.nav_bar.updateGeometry()
        
        # 计算滚动区域应该占据的空间
        nav_bottom = current_geo.y() + current_geo.height()
        scroll_top = nav_bottom + 16  # 导航栏下方留16px间距
        scroll_height = page_height - scroll_top - padding  # 底部留padding间距
        
        # 直接设置scroll_area的位置和大小
        if hasattr(self, 'scroll_area'):
            self.scroll_area.setGeometry(
                0,
                scroll_top,
                page_width,
                scroll_height
            )
    
    def _create_model_option_card(self, model_info: dict) -> SiOptionCardLinear:
        """
        创建模型选项卡

        Args:
            model_info: 模型信息字典（来自 _scan_models 或 on_model_created）

        Returns:
            SiOptionCardLinear: 配置好的模型选项卡
        """
        name = model_info.get('name', '?')
        model_type = model_info.get('type', 'SAEHD')
        class_name = model_info.get('class_name', model_type)
        precision = model_info.get('precision', 'fp32')
        resolution = model_info.get('resolution', '?')
        archi = model_info.get('archi', '?')
        ae_dims = model_info.get('ae_dims', '?')
        e_dims = model_info.get('e_dims', '?')
        d_dims = model_info.get('d_dims', '?')
        d_mask_dims = model_info.get('d_mask_dims', '?')
        face_type = model_info.get('face_type', '?')
        batch_size = model_info.get('batch_size', '?')
        lr = model_info.get('lr', '?')
        gan_power = model_info.get('gan_power', '?')
        training_iter = model_info.get('iter', 0)

        subtitle = (f"{resolution} | {face_type} | {archi} | ae{ae_dims} e{e_dims} d{d_dims} dm{d_mask_dims}"
                    f" | {precision} | bs{batch_size} | iter {training_iter}")

        option_card = SiOptionCardLinear(self.models_card)
        option_card.setTitle(name, subtitle)
        option_card.load(safe_get_icon("ic_fluent_box_multiple_filled"))

        # 构建详细工具提示
        def _v(key, label):
            v = model_info.get(key, '?')
            return f"{label}: {v}"

        tooltip_lines = [
            f"{name} ({class_name})",
            f"分辨率: {resolution} | 人脸: {face_type} | 架构: {archi}",
            f"AE: {ae_dims}  E: {e_dims}  D: {d_dims}  DM: {d_mask_dims}",
            f"精度: {precision} | 批次: {batch_size}",
        ]
        extra_params = [
            ('lr', '学习率'), ('lr_cos', '余弦周期'), ('lr_dropout', 'LR策略'),
            ('adabelief', 'AdaBelief'), ('clipgrad', '梯度裁剪'),
            ('gan_power', 'GAN强度'), ('gan_patch_size', 'GAN块'), ('gan_dims', 'GAN维度'),
            ('true_face_power', '真脸强度'), ('face_style_power', '面部风格'), ('bg_style_power', '背景风格'),
            ('random_warp', '随机形变'), ('random_hsv_power', 'HSV强度'), ('ct_mode', '色彩迁移'),
            ('eyes_mouth_prio', '嘴眼优先'), ('uniform_yaw', '均匀偏航'), ('blur_out_mask', '遮罩羽化'),
            ('masked_training', '遮罩训练'), ('pretrain', '预训练'),
            ('models_opt_on_gpu', '优化器GPU'), ('use_fast_generator', '快速生成器'),
            ('gradient_checkpointing', '梯度检查点'),
            ('crash_threshold', '崩溃阈值'), ('backup_interval', '备份间隔'), ('max_backups', '最大备份'),
            ('target_iter', '目标迭代'), ('write_preview_history', '预览历史'),
        ]
        for key, label in extra_params:
            v = model_info.get(key)
            if v != '?':
                tooltip_lines.append(f"  {label}: {v}")
        tooltip_lines.append(f"当前迭代: {training_iter}")
        option_card.setToolTip('\n'.join(tooltip_lines))
        
        train_button = SiPushButtonRefactor(option_card)
        train_button.setText("训练")
        train_button.setSvgIcon(safe_get_icon("ic_fluent_caret_right_regular"))
        train_button.adjustSize()
        # 绑定点击事件，打开训练配置子页面
        train_button.clicked.connect(lambda: self.open_training_config(name))
        
        option_card.addWidget(train_button)
        option_card.adjustSize()
        
        # 使用QGraphicsOpacityEffect实现淡入效果
        opacity_effect = QGraphicsOpacityEffect(option_card)
        opacity_effect.setOpacity(0.0)
        option_card.setGraphicsEffect(opacity_effect)
        
        # 使用QVariantAnimation进行动画
        fade_animation = QVariantAnimation(option_card)
        fade_animation.setStartValue(0.0)
        fade_animation.setEndValue(1.0)
        fade_animation.setDuration(300)
        fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 连接值变化信号
        def update_opacity(value):
            opacity_effect.setOpacity(float(value))
        
        fade_animation.valueChanged.connect(update_opacity)
        fade_animation.start()
        
        # 保存动画引用，防止被垃圾回收
        if not hasattr(self, '_fade_animations'):
            self._fade_animations = []
        self._fade_animations.append(fade_animation)
        
        return option_card
    
    def on_new_model_clicked(self):
        """新建模型按钮点击事件"""
        # 生成新模型名称
        new_model_name = f"model_{self.model_counter:03d}"
        self.model_counter += 1
        
        
        # 打开新建模型配置子页面
        self.open_new_model_config(new_model_name)
        
    
    def _animate_models_card_height(self):
        """动画调整Models卡片高度"""
        if not hasattr(self, 'models_card'):
            return
        
        card = self.models_card
        current_width = card.width()
        current_height = card.height()
        
        
        # 不要取消宽度限制，直接计算目标高度
        # 强制刷新布局
        if card.layout():
            card.layout().invalidate()
            card.layout().activate()
        QApplication.processEvents()
        
        # 延迟计算目标高度
        QTimer.singleShot(10, lambda: self._calculate_and_animate_card(card, current_width, current_height))
    
    def _calculate_and_animate_card(self, card, current_width, current_height):
        """延迟计算目标高度并启动动画"""
        # 保持当前宽度，不要让它变化
        card.setFixedWidth(current_width)
        
        # 计算目标高度
        card.adjustSize()
        target_height = card.sizeHint().height()
        
        
        # 如果目标高度与当前高度相同，不需要动画
        if abs(target_height - current_height) < 1:
            return
        
        # 创建或获取高度动画
        card_id = id(card)
        if card_id not in self.card_height_animations:
            animation = SiExpAnimation(card)
            animation.setFactor(1/8)
            animation.setBias(0.2)
            # 在动画的每一帧都更新卡片高度，同时保持宽度
            def update_height(value):
                card.setFixedHeight(int(value))
                card.setFixedWidth(current_width)  # 保持宽度不变
            animation.ticked.connect(update_height)
            self.card_height_animations[card_id] = animation
        
        # 设置动画的起始和目标值
        animation = self.card_height_animations[card_id]
        animation.setCurrent(current_height)
        animation.setTarget(target_height)
        animation.start()
    
    def update_scroll_area_after_resize(self):
        """卡片高度动画结束后更新滚动区域"""
        if not hasattr(self, 'scroll_area'):
            return
        
        # 获取当前页面尺寸
        page_height = self.height()
        padding = self.padding
        
        # 重新计算导航栏位置
        current_geo = self.nav_bar.geometry()
        nav_bottom = current_geo.y() + current_geo.height()
        scroll_top = nav_bottom + 16
        scroll_height = page_height - scroll_top - padding
        
        # 更新scroll_area
        self.scroll_area.setGeometry(
            0,
            scroll_top,
            self.width(),
            scroll_height
        )
        
        # 关键：更新titled_widgets_group的最小高度，让滚动区域知道内容变大了
        if hasattr(self, 'titled_widgets_group'):
            content_height = self.titled_widgets_group.sizeHint().height()
            self.titled_widgets_group.setMinimumHeight(content_height)
        
        pass
    
    def check_models_card_height(self):
        """定时器回调：检查Models卡片高度是否变化"""
        if not hasattr(self, 'models_card') or not hasattr(self, 'last_models_card_height'):
            return
        
        card = self.models_card
        current_height = card.height()
        current_width = card.width()
        
        # 如果高度发生变化，强制刷新卡片和父容器
        if abs(current_height - self.last_models_card_height) > 1:
            
            # 强制刷新卡片尺寸
            card.adjustSize()
            card.updateGeometry()
            card.update()
            
            # 刷新父容器
            if hasattr(card, 'parentWidget') and card.parentWidget():
                parent = card.parentWidget()
                parent.updateGeometry()
                parent.update()
            
            # 更新scroll_area
            self.update_scroll_area_after_resize()
            
            QApplication.processEvents()
            
            self.last_models_card_height = current_height
    
    @staticmethod
    def _scan_models(model_dir):
        """扫描模型目录中的已保存模型实例，从 _data.dat 中读取真实配置"""
        import os
        import pickle
        import io
        from datetime import datetime

        # 受限 unpickler：跳过 torch 张量（UI 中 torch DLL 可能无法加载）
        class _TorchStub:
            """替换 torch 类型的占位符，接受任意 pickle 操作。"""
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)
            def __init__(self, *args, **kwargs):
                pass
            def __setstate__(self, state):
                pass

        class _ModelDataUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch' or module.startswith('torch.'):
                    return _TorchStub
                return super().find_class(module, name)

        results = []
        base = Path(model_dir)
        if not base.exists():
            return results
        seen = set()
        for f in base.iterdir():
            if not f.is_file() or not f.name.endswith('_data.dat'):
                continue
            stem = f.name[:-len('_data.dat')]
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                continue
            base_name, class_name = parts
            if base_name in seen:
                continue
            seen.add(base_name)

            options = {}
            training_iter = 0
            try:
                model_data = _ModelDataUnpickler(io.BytesIO(f.read_bytes())).load()
                options = model_data.get('options', {})
                # 只移除可能泄漏的 _TorchStub 占位符，保留 numpy 等合法类型
                options = {k: v for k, v in options.items()
                           if type(v).__name__ != '_TorchStub'}
                training_iter = model_data.get('iter', 0)
            except Exception as e:
                import traceback
                print(f"\n[ERROR] 反序列化模型数据失败 ({f.name}): {e}")
                traceback.print_exc()

            precision = 'bf16' if options.get('use_bf16', False) else 'fp32'
            archi_val = options.get('archi', '')
            archi_type = archi_val.split('-')[0] if archi_val else ''

            results.append({
                'name': base_name,
                'type': archi_type.upper() if archi_type else class_name,
                'class_name': class_name,
                'precision': precision,
                'use_bf16': options.get('use_bf16', False),
                'resolution': options.get('resolution', '?'),
                'archi': archi_val or '?',
                'ae_dims': options.get('ae_dims', '?'),
                'e_dims': options.get('e_dims', '?'),
                'd_dims': options.get('d_dims', '?'),
                'd_mask_dims': options.get('d_mask_dims', '?'),
                'face_type': options.get('face_type', '?'),
                'batch_size': options.get('batch_size', '?'),
                'lr': options.get('lr', '?'),
                'lr_cos': options.get('lr_cos', '?'),
                'lr_dropout': options.get('lr_dropout', '?'),
                'adabelief': options.get('adabelief', '?'),
                'clipgrad': options.get('clipgrad', '?'),
                'gan_power': options.get('gan_power', '?'),
                'gan_patch_size': options.get('gan_patch_size', '?'),
                'gan_dims': options.get('gan_dims', '?'),
                'true_face_power': options.get('true_face_power', '?'),
                'face_style_power': options.get('face_style_power', '?'),
                'bg_style_power': options.get('bg_style_power', '?'),
                'random_warp': options.get('random_warp', '?'),
                'random_hsv_power': options.get('random_hsv_power', '?'),
                'ct_mode': options.get('ct_mode', '?'),
                'eyes_mouth_prio': options.get('eyes_mouth_prio', '?'),
                'uniform_yaw': options.get('uniform_yaw', '?'),
                'blur_out_mask': options.get('blur_out_mask', '?'),
                'masked_training': options.get('masked_training', '?'),
                'pretrain': options.get('pretrain', '?'),
                'models_opt_on_gpu': options.get('models_opt_on_gpu', '?'),
                'use_fast_generator': options.get('use_fast_generator', '?'),
                'gradient_checkpointing': options.get('gradient_checkpointing', '?'),
                'crash_threshold': options.get('crash_threshold', '?'),
                'backup_interval': options.get('backup_interval', '?'),
                'max_backups': options.get('max_backups', '?'),
                'target_iter': options.get('target_iter', '?'),
                'write_preview_history': options.get('write_preview_history', '?'),
                'random_src_flip': options.get('random_src_flip', '?'),
                'random_dst_flip': options.get('random_dst_flip', '?'),
                'freeze_encoder': options.get('freeze_encoder', False),
                'freeze_decoder_mask': options.get('freeze_decoder_mask', False),
                'freeze_inter': options.get('freeze_inter', False),
                'freeze_inter_AB': options.get('freeze_inter_AB', False),
                'freeze_inter_B': options.get('freeze_inter_B', False),
                'freeze_inter_src': options.get('freeze_inter_src', False),
                'freeze_inter_dst': options.get('freeze_inter_dst', False),
                'iter': training_iter,
                'mtime': datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        results.sort(key=lambda x: x['mtime'], reverse=True)
        return results

    def _on_model_dir_changed(self, new_path):
        """模型目录路径变化时重新扫描"""
        self._model_dir = new_path
        self._rebuild_model_list(new_path)

    def _rebuild_model_list(self, model_dir):
        """清空并重建模型列表卡片"""
        layout = self.models_container.layout()
        # Remove old cards (keep the button if present — it's the last widget)
        for i in range(layout.count() - 1, -1, -1):
            w = layout.itemAt(i).widget()
            if w and isinstance(w, (SiOptionCardLinear, QLabel)):
                layout.removeWidget(w)
                w.deleteLater()

        models = self._scan_models(model_dir)

        # 存储模型信息以便后续查找（例如打开配置页面时）
        self._models_info = {m['name']: m for m in models}

        if not models:
            # 显示空状态提示
            empty_label = QLabel("暂无已保存的模型，点击下方按钮新建")
            empty_label.setStyleSheet("color: #888; padding: 12px 0;")
            layout.insertWidget(0, empty_label)
        else:
            for i, m in enumerate(models):
                option_card = self._create_model_option_card(m)
                layout.insertWidget(i, option_card)

    def open_training_config(self, model_name: str):
        """打开训练配置子页面（点击训练按钮时）"""
        model_info = self._models_info.get(model_name, {})
        # 获取主窗口
        main_window = SiGlobal.siui.windows.get("MAIN_WINDOW")
        if main_window and hasattr(main_window, 'layerChildPage'):
            # 创建并显示子页面（传递模型信息和WebUI设置）
            child_page = TrainingConfigChildPage(self, model_name, model_info=model_info, model_dir=self._model_dir,
                                                  webui_port=self._webui_port, webui_password=self._webui_password)
            main_window.layerChildPage().setChildPage(child_page)
        else:
            print(f"[WARNING] 无法找到主窗口或layerChildPage")
    
    def open_new_model_config(self, model_name: str):
        """打开新建模型配置子页面（点击新建模型按钮时）"""
        model_info = self._models_info.get(model_name, {})
        # 获取主窗口
        main_window = SiGlobal.siui.windows.get("MAIN_WINDOW")
        if main_window and hasattr(main_window, 'layerChildPage'):
            # 创建并显示子页面（传递模型信息）
            child_page = NewModelConfigChildPage(self, model_name, model_info=model_info)
            
            # 连接信号：当用户点击创建模型时
            child_page.model_created.connect(self.on_model_created)
            
            main_window.layerChildPage().setChildPage(child_page)
        else:
            print(f"[WARNING] 无法找到主窗口或layerChildPage")
    
    def on_model_created(self, config_data: dict):
        """处理模型创建事件"""
        
        # 从配置中提取信息
        model_name = config_data.get('model_name', 'unknown')
        archi = config_data.get('archi', 'DF')
        subarchi = config_data.get('subarchi', '-ud')
        resolution = config_data.get('resolution', '256')
        ae_dims = config_data.get('ae_dims', '256')
        e_dims = config_data.get('e_dims', '64')
        d_dims = config_data.get('d_dims', '128')
        d_mask_dims = config_data.get('d_mask_dims', '32')
        
        # 组合架构字符串
        full_archi = f"{archi.lower()}{subarchi}"
        
        # 创建新的模型选项卡（包含淡入动画）
        option_card = self._create_model_option_card({
            'name': model_name,
            'type': archi.upper(),
            'class_name': 'SAEHD',
            'precision': 'fp32',
            'resolution': resolution,
            'archi': full_archi.lower(),
            'ae_dims': ae_dims,
            'e_dims': e_dims,
            'd_dims': d_dims,
            'd_mask_dims': d_mask_dims,
            'face_type': 'wf',
            'batch_size': config_data.get('batch_size', '?'),
            'lr': config_data.get('lr', '?'),
            'iter': 0,
        })
        
        # 添加到models_container中（在“新建模型”按钮之前）
        # 找到“新建模型”按钮的索引
        new_model_button_index = -1
        for i in range(self.models_container.layout().count()):
            widget = self.models_container.layout().itemAt(i).widget()
            if widget and hasattr(widget, 'text') and widget.text() == "新建模型":
                new_model_button_index = i
                break
        
        if new_model_button_index > 0:
            # 在按钮之前插入新卡片
            self.models_container.layout().insertWidget(new_model_button_index, option_card)
        else:
            # 如果没找到按钮，直接添加到最后
            self.models_container.addWidget(option_card)
        

        # 触发卡片高度动画（会自动处理布局刷新）
        self._animate_models_card_height()

