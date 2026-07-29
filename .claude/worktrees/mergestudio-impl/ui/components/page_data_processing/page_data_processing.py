from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect, QSizePolicy, QBoxLayout
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from siui.components import SiDenseHContainer, SiDenseVContainer, SiTitledWidgetGroup
from siui.components.button import SiPushButtonRefactor
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.container import SiTriSectionPanelCard
from siui.components.editbox import SiLabeledLineEdit
from siui.components.page import SiPage
from siui.components.widgets import SiCheckBox, SiLabel
from siui.core import SiExpAnimation
@contextmanager
def createPanelCard(parent: SiTitledWidgetGroup, title: str) -> SiTriSectionPanelCard:
    """创建面板卡片的上下文管理器"""
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    # 设置卡片为自适应大小
    card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
    try:
        yield card
    finally:
        # 确保内容完全添加后再调整大小
        card.adjustSize()
        parent.addWidget(card)
@contextmanager
def createDenseContainer(parent,
                         direction: QBoxLayout.Direction,
                         side: Qt.Edges = Qt.LeftEdge | Qt.TopEdge):
    """创建密集容器的上下文管理器"""
    from siui.components.container import SiDenseContainer
    container = SiDenseContainer(parent)
    container.layout().setDirection(direction)
    container.layout().setSpacing(12)
    # 设置容器为自适应大小，允许垂直扩展
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
    try:
        yield container
    finally:
        parent.addWidget(container, side)
class DataProcessingPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("数据处理")
        
        # 初始化卡片高度动画字典
        self.card_height_animations = {}
        # 创建滚动容器
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)
        # 第一组：Analyzer - 元数据分析
        with self.titled_widgets_group as group:
            group.addTitle("元数据分析器 (Analyzer)")
            with createPanelCard(group, "分析设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.analyzer_input_path_input = SiLabeledLineEdit(self)
                    self.analyzer_input_path_input.setTitle("人脸集路径")
                    self.analyzer_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.analyzer_input_path_input.setText(str(default_input))
                    self.analyzer_input_path_input.resize(700, 64)
                    container.addWidget(self.analyzer_input_path_input)
                    
                    # 特征选择区域（使用 siui 容器）
                    self.feature_selection_container = SiDenseVContainer(self)
                    self.feature_selection_container.setSpacing(12)
                    self.feature_selection_container.setAlignment(Qt.AlignLeft)
                    self.feature_selection_container.setMinimumHeight(350)  # 设置最小高度以容纳所有复选框和按钮
                    
                    # 标题标签
                    self.feature_title_label = SiLabel(self)
                    self.feature_title_label.setText("选择分析特征")
                    self.feature_title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
                    self.feature_selection_container.addWidget(self.feature_title_label)
                    
                    # 创建特征复选框（使用 siui 的 SiCheckBox）
                    self.feature_checkboxes = {}
                    feature_options = [
                        ("phash", "感知哈希 (phash) - 用于重复检测"),
                        ("histogram", "直方图分布 (histogram) - RGB和HSV统计"),
                        ("hue", "色相分析 (hue) - 颜色分布统计"),
                        ("pose", "人脸姿态 (pose) - 需要landmarks"),
                        ("embedding", "人脸Embedding (embedding) - ArcFace特征向量"),
                        ("landmark", "人脸特征点 (landmark) - InsightFace 106pt检测"),
                    ]
                    
                    for feature_key, feature_desc in feature_options:
                        checkbox = SiCheckBox(self)
                        checkbox.setText(feature_desc)
                        checkbox.setChecked(True)  # 默认全选
                        checkbox.resize(400, 32)
                        checkbox.setToolTip(f"启用 {feature_key} 特征分析")
                        self.feature_checkboxes[feature_key] = checkbox
                        self.feature_selection_container.addWidget(checkbox)
                    
                    # 全选/取消全选按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as btn_row:
                        self.select_all_button = SiPushButtonRefactor(self)
                        self.select_all_button.setText("全选")
                        self.select_all_button.resize(120, 36)
                        self.select_all_button.clicked.connect(self.on_select_all_features)
                        
                        self.deselect_all_button = SiPushButtonRefactor(self)
                        self.deselect_all_button.setText("取消全选")
                        self.deselect_all_button.resize(120, 36)
                        self.deselect_all_button.clicked.connect(self.on_deselect_all_features)
                        
                        btn_row.addWidget(self.select_all_button)
                        btn_row.addWidget(self.deselect_all_button)
                    
                    self.feature_selection_container.addWidget(btn_row)
                    container.addWidget(self.feature_selection_container)
            # 运行控制卡片
            with createPanelCard(group, "运行控制") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container:
                    # 强制重新分析开关（暂时用按钮代替）
                    self.analyzer_force_button = SiPushButtonRefactor(self)
                    self.analyzer_force_button.setText("强制重新分析")
                    self.analyzer_force_button.resize(180, 48)
                    self.analyzer_force_button.setToolTip("覆盖已有数据，重新计算所有特征")
                    self.analyzer_force_button.clicked.connect(lambda: self.on_run_analyzer_clicked(force_reanalyze=True))  # 绑定到相同的处理函数，但传递force=True
                    
                    # 写回源文件按钮
                    self.write_back_button = SiPushButtonRefactor(self)
                    self.write_back_button.setText("写回源文件")
                    self.write_back_button.resize(180, 48)
                    self.write_back_button.setToolTip("用于兼容原版DeepFaceLab，虽然我不知道为什么你还要用回老版本")
                    self.write_back_button.clicked.connect(self.on_write_back_clicked)
                    
                    # 运行按钮
                    self.run_analyzer_button = SiPushButtonRefactor(self)
                    self.run_analyzer_button.setText("开始分析")
                    self.run_analyzer_button.resize(180, 48)
                    self.run_analyzer_button.setToolTip("点击开始执行元数据分析任务")
                    self.run_analyzer_button.clicked.connect(self.on_run_analyzer_clicked)
                    
                    container.addWidget(self.analyzer_force_button)
                    container.addWidget(self.write_back_button)
                    container.addWidget(self.run_analyzer_button)
        # 第二组：Filter - 人脸集过滤
        with self.titled_widgets_group as group:
            group.addTitle("人脸集过滤器 (Filter)")
            with createPanelCard(group, "过滤设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    container.layout().setSpacing(16)  # 增加行间距
                    
                    # 输入路径输入框
                    self.filter_input_path_input = SiLabeledLineEdit(self)
                    self.filter_input_path_input.setTitle("人脸集路径")
                    self.filter_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.filter_input_path_input.setText(str(default_input))
                    self.filter_input_path_input.resize(700, 64)
                    container.addWidget(self.filter_input_path_input)
            # 过滤模式和控制卡片
            with createPanelCard(group, "过滤模式") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：过滤模式选择
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        self.filter_mode_combobox = SiCapsuleComboBox(self)
                        self.filter_mode_combobox.setTitle("过滤模式")
                        self.filter_mode_combobox.setMinimumHeight(36)
                        self.filter_mode_combobox.setEditable(False)
                        self.filter_mode_combobox.addItems([
                            "blur",         # 模糊过滤（拉普拉斯方差）
                            "faceid",       # 人脸ID分组
                            "position",     # 位置聚类分组
                            "repeated"      # 重复图片去重
                        ])
                        self.filter_mode_combobox.setCurrentText("blur")
                        self.filter_mode_combobox.setToolTip("选择过滤模式：模糊过滤/人脸ID分组/位置聚类/重复去重")
                        self.filter_mode_combobox.currentTextChanged.connect(self.on_filter_mode_changed)
                        
                        row1.addWidget(self.filter_mode_combobox)
                    # 第二行：位置过滤参数（仅 position 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_params:
                        self.filter_position_eps_input = SiLabeledLineEdit(self)
                        self.filter_position_eps_input.setTitle("邻域半径 (像素)")
                        self.filter_position_eps_input.setPlaceholderText("默认: 50")
                        self.filter_position_eps_input.setText("50")
                        self.filter_position_eps_input.resize(180, 48)
                        self.filter_position_eps_input.setToolTip("DBSCAN 聚类半径，位置距离小于此值的人脸视为同一位置")
                        self.filter_position_eps_input.setVisible(False)
                        
                        self.filter_position_min_samples_input = SiLabeledLineEdit(self)
                        self.filter_position_min_samples_input.setTitle("最小样本数")
                        self.filter_position_min_samples_input.setPlaceholderText("默认: 2")
                        self.filter_position_min_samples_input.setText("2")
                        self.filter_position_min_samples_input.resize(150, 48)
                        self.filter_position_min_samples_input.setToolTip("形成一组所需的最少人脸数量")
                        self.filter_position_min_samples_input.setVisible(False)
                        
                        row_params.addWidget(self.filter_position_eps_input)
                        row_params.addWidget(self.filter_position_min_samples_input)
                    # 第三行：重复去重参数（仅 repeated 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_repeated_params:
                        self.filter_repeated_threshold_input = SiLabeledLineEdit(self)
                        self.filter_repeated_threshold_input.setTitle("相似度阈值")
                        self.filter_repeated_threshold_input.setPlaceholderText("默认: 0.98")
                        self.filter_repeated_threshold_input.setText("0.98")
                        self.filter_repeated_threshold_input.resize(180, 48)
                        self.filter_repeated_threshold_input.setToolTip("图片相似度阈值 (0.0-1.0)，高于此值视为重复（使用phash感知哈希）")
                        self.filter_repeated_threshold_input.setVisible(False)
                        
                        row_repeated_params.addWidget(self.filter_repeated_threshold_input)
                    # 第四行：人脸ID过滤参数（仅 faceid 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_faceid_params:
                        self.filter_faceid_eps_input = SiLabeledLineEdit(self)
                        self.filter_faceid_eps_input.setTitle("聚类半径 (eps)")
                        self.filter_faceid_eps_input.setPlaceholderText("默认: 0.114514")
                        self.filter_faceid_eps_input.setText("0.114514")
                        self.filter_faceid_eps_input.resize(180, 48)
                        self.filter_faceid_eps_input.setToolTip("DBSCAN 聚类半径（余弦距离空间），值越小分组越严格")
                        self.filter_faceid_eps_input.setVisible(False)
                        
                        self.filter_faceid_min_samples_input = SiLabeledLineEdit(self)
                        self.filter_faceid_min_samples_input.setTitle("最小样本数")
                        self.filter_faceid_min_samples_input.setPlaceholderText("默认: 2")
                        self.filter_faceid_min_samples_input.setText("2")
                        self.filter_faceid_min_samples_input.resize(150, 48)
                        self.filter_faceid_min_samples_input.setToolTip("形成一组所需的最少人脸数量")
                        self.filter_faceid_min_samples_input.setVisible(False)
                        
                        row_faceid_params.addWidget(self.filter_faceid_eps_input)
                        row_faceid_params.addWidget(self.filter_faceid_min_samples_input)
                    # 第五行：运行按钮 + 合并按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row2:
                        self.run_filter_button = SiPushButtonRefactor(self)
                        self.run_filter_button.setText("开始过滤")
                        self.run_filter_button.resize(180, 48)
                        self.run_filter_button.setToolTip("点击开始执行人脸集过滤任务")
                        self.run_filter_button.clicked.connect(self.on_run_filter_clicked)
                        
                        self.merge_subfolders_button = SiPushButtonRefactor(self)
                        self.merge_subfolders_button.setText("合并子文件夹")
                        self.merge_subfolders_button.resize(180, 48)
                        self.merge_subfolders_button.setToolTip("将所有一级子文件夹中的图片合并到 aligned 目录（安全：不递归）")
                        self.merge_subfolders_button.clicked.connect(self.on_merge_subfolders_clicked)
                        
                        row2.addWidget(self.run_filter_button)
                        row2.addWidget(self.merge_subfolders_button)
        # 第三组：Sorter - 智能排序
        with self.titled_widgets_group as group:
            group.addTitle("智能排序器 (Sorter)")
            with createPanelCard(group, "排序设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.sorter_input_path_input = SiLabeledLineEdit(self)
                    self.sorter_input_path_input.setTitle("人脸集路径")
                    self.sorter_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.sorter_input_path_input.setText(str(default_input))
                    self.sorter_input_path_input.resize(700, 64)
                    container.addWidget(self.sorter_input_path_input)
            # 排序模式和控制卡片
            with createPanelCard(group, "排序模式") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：排序方式选择 + 姿态类型（同一行）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        self.sorter_method_combobox = SiCapsuleComboBox(self)
                        self.sorter_method_combobox.setTitle("排序方式")
                        self.sorter_method_combobox.setMinimumHeight(36)
                        self.sorter_method_combobox.setEditable(False)
                        self.sorter_method_combobox.addItems([
                            "phash",              # 感知哈希排序
                            "hist",               # 直方图排序
                            "blur",               # 模糊度排序
                            "face_pose",          # 人脸姿态排序
                            "resolution",         # 分辨率排序
                            "color",              # 颜色排序
                            "name",               # 文件名排序
                        ])
                        self.sorter_method_combobox.setCurrentText("phash")
                        self.sorter_method_combobox.setToolTip("选择排序方式")
                        self.sorter_method_combobox.currentTextChanged.connect(self.on_sorter_method_changed)
                        
                        # 姿态类型选择（仅 face_pose 时显示）
                        self.sorter_pose_type_combobox = SiCapsuleComboBox(self)
                        self.sorter_pose_type_combobox.setTitle("姿态类型")
                        self.sorter_pose_type_combobox.setMinimumHeight(36)
                        self.sorter_pose_type_combobox.setEditable(False)
                        self.sorter_pose_type_combobox.addItems(["pitch", "yaw", "roll"])
                        self.sorter_pose_type_combobox.setCurrentText("yaw")
                        self.sorter_pose_type_combobox.setToolTip("选择姿态角度类型（仅用于face_pose排序）")
                        self.sorter_pose_type_combobox.setVisible(False)
                        
                        row1.addWidget(self.sorter_method_combobox)
                        row1.addWidget(self.sorter_pose_type_combobox)
                    # 第二行：运动模糊检测开关（仅 blur 模式显示）
                    self.sorter_motion_blur_checkbox = SiCheckBox(self)
                    self.sorter_motion_blur_checkbox.setText("使用运动模糊检测")
                    self.sorter_motion_blur_checkbox.setMinimumHeight(36)
                    self.sorter_motion_blur_checkbox.setChecked(False)
                    self.sorter_motion_blur_checkbox.setToolTip("启用运动模糊检测（仅用于blur排序）")
                    self.sorter_motion_blur_checkbox.setVisible(False)
                    container.addWidget(self.sorter_motion_blur_checkbox)
                    # 第三行：运行按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row2:
                        self.run_sorter_button = SiPushButtonRefactor(self)
                        self.run_sorter_button.setText("开始排序")
                        self.run_sorter_button.resize(180, 48)
                        self.run_sorter_button.setToolTip("点击开始执行人脸集排序任务")
                        self.run_sorter_button.clicked.connect(self.on_run_sorter_clicked)
                        
                        row2.addWidget(self.run_sorter_button)
            # 初始化时确保控件状态正确（不触发动画）
            self.sorter_pose_type_combobox.setVisible(False)
            self.sorter_motion_blur_checkbox.setVisible(False)
        # 添加页脚空白
        self.titled_widgets_group.addPlaceholder(64)
        # 设置为页面对象
        self.setAttachment(self.titled_widgets_group)
    def _create_task_monitor(self, button, completed_description: str, interrupted_description: str = "用户强制中断了一个任务"):
        """
        创建任务监控器的工厂方法
        
        Args:
            button: 需要监控的按钮对象
            completed_description: 完成时的描述文本
            interrupted_description: 中断时的描述文本
            
        Returns:
            monitor_function: 监控函数
        """
        original_text = button.text()
        
        def monitor_process(process):
            """监控进程并更新UI"""
            try:
                process.wait()  # 等待进程结束
                # 正常完成 - 显示绿色通知
                parent_window = self.window()
                if hasattr(parent_window, 'show_task_completed_notification'):
                    parent_window.show_task_completed_notification(completed_description)
                print(f"✓ {completed_description}")
            except Exception as e:
                # 异常结束
                error_msg = str(e)
                parent_window = self.window()
                if hasattr(parent_window, 'show_task_interrupted_notification'):
                    if 'terminated' in error_msg.lower() or 'killed' in error_msg.lower():
                        parent_window.show_task_interrupted_notification(interrupted_description)
                    else:
                        parent_window.show_task_error_notification(f"任务异常结束: {error_msg}")
                print(f"✗ 任务异常: {error_msg}")
            finally:
                # 恢复按钮状态
                button.setText(original_text)
                button.setEnabled(True)
        
        return monitor_process
    def on_select_all_features(self):
        """全选所有特征"""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)
    
    def on_deselect_all_features(self):
        """取消全选所有特征"""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_features(self):
        """获取用户选择的特征列表"""
        selected = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]
        return selected
    
    def fade_widget_in(self, widget, duration: int = 200):
        """
        淡入显示控件
        """
                        
        widget.setVisible(True)
        opacity_effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(opacity_effect)
        
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()
        
        # 保存引用防止被回收
        if not hasattr(self, '_fade_animations'):
            self._fade_animations = []
        self._fade_animations.append(animation)
    
    def fade_widget_out(self, widget, duration: int = 200):
        """
        淡出隐藏控件
        """
                        
        opacity_effect = widget.graphicsEffect()
        if opacity_effect is None:
            opacity_effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(opacity_effect)
        
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 动画结束后隐藏
        animation.finished.connect(lambda: widget.setVisible(False))
        animation.start()
        
        # 保存引用
        if not hasattr(self, '_fade_animations'):
            self._fade_animations = []
        self._fade_animations.append(animation)
    
    def on_sorter_method_changed(self, method: str):
        """排序方式改变时的处理"""
        # 根据排序方法显示/隐藏相关控件（带动画）
        if method == "face_pose":
            # 显示姿态类型选择（同一行，不触发动画）
            self.fade_widget_in(self.sorter_pose_type_combobox)
            self.fade_widget_out(self.sorter_motion_blur_checkbox)
        elif method == "blur":
            # 显示运动模糊选项（第二行，需要触发高度动画）
            self.fade_widget_out(self.sorter_pose_type_combobox)
            self.fade_widget_in(self.sorter_motion_blur_checkbox)
        else:
            # 隐藏特殊选项
            self.fade_widget_out(self.sorter_pose_type_combobox)
            self.fade_widget_out(self.sorter_motion_blur_checkbox)
        
        # 只有当 motion_blur_checkbox 的可见性真正改变时才触发动画
        # 使用定时器等待 fade 动画完成后再调整卡片高度
        QTimer.singleShot(250, lambda: self._animate_sorter_card_height(method))
    
    def _animate_sorter_card_height(self, method: str):
        """在 fade 动画完成后调整卡片高度"""
        if not hasattr(self, 'sorter_method_combobox'):
            return
        
        # 找到包含排序模式的卡片
        card = self.sorter_method_combobox.parent()
        while card and not hasattr(card, 'setTitle'):
            card = card.parent()
        
        if not card:
            return
        
        # 保存当前宽度
        current_width = card.width()
        current_height = card.height()
        
        # 临时取消固定宽度限制
        card.setMinimumWidth(0)
        card.setMaximumWidth(16777215)
        
        # 强制刷新布局
        card.layout().invalidate()
        card.layout().activate()
        QApplication.processEvents()
        
        # 延迟计算目标高度
        QTimer.singleShot(10, lambda: self._calculate_sorter_card_animation(card, current_width, current_height))
    
    def _calculate_sorter_card_animation(self, card, current_width, current_height):
        """延迟计算排序卡片目标高度并启动动画"""
        # 计算目标高度
        card.adjustSize()
        target_height = card.sizeHint().height()
        
        # 恢复宽度限制并设置当前宽度
        card.setFixedWidth(current_width)
        
        # 如果目标高度与当前高度相同，不需要动画
        if abs(target_height - current_height) < 1:
            return
        
        # 创建或获取高度动画
        sorter_animation_key = f"sorter_{id(card)}"
        if sorter_animation_key not in self.card_height_animations:
            animation = SiExpAnimation(card)
            animation.setFactor(1/8)
            animation.setBias(0.2)
            animation.ticked.connect(lambda value, c=card: c.setFixedHeight(int(value)))
            self.card_height_animations[sorter_animation_key] = animation
        
        # 设置动画的起始和目标值
        animation = self.card_height_animations[sorter_animation_key]
        animation.setCurrent(current_height)
        animation.setTarget(target_height)
        animation.start()
    
    def on_write_back_clicked(self):
        """写回源文件按钮点击事件"""
        # 获取输入路径
        input_path = self.analyzer_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 构建命令
        analyzer_script = project_root / "FacesetProcessor" / "Analyzer.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(analyzer_script),
            "--input", str(input_path_obj),
            "--write-back",
        ]
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print("提示: 用于兼容原版DeepFaceLab，虽然我不知道为什么你还要用回老版本")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                "写回源文件 - DFL兼容模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.write_back_button,
                "元数据写回已完成",
                "用户强制中断了元数据写回任务"
            )
            
            # 设置按钮为运行状态
            self.write_back_button.setText("正在写入...")
            self.write_back_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动元数据写回进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.write_back_button.setText("写回源文件")
            self.write_back_button.setEnabled(True)
    
    def on_run_analyzer_clicked(self, force_reanalyze: bool = False):
        """Analyzer 运行按钮点击事件
        
        Args:
            force_reanalyze: 是否强制重新分析（覆盖已有数据）
        """
        # 获取输入路径
        input_path = self.analyzer_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 获取选择的特征
        selected_features = self.get_selected_features()
        if not selected_features:
            print("错误: 请至少选择一个分析特征")
            return
        
        # 构建命令
        analyzer_script = project_root / "FacesetProcessor" / "Analyzer.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(analyzer_script),
            "--input", str(input_path_obj),
            "--features", ",".join(selected_features),
        ]
        
        # 如果勾选了强制重新分析，添加 --force 参数
        if force_reanalyze:
            cmd.append("--force")
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print(f"已选择的特征: {', '.join(selected_features)}")
        print(f"强制重新分析: {'是' if force_reanalyze else '否'}")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            mode_text = "强制重新分析" if force_reanalyze else "增量分析"
            parent_window.show_command_notification(
                cmd_str,
                f"元数据分析 - {mode_text} - 特征: {', '.join(selected_features)}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_analyzer_button,
                f"元数据分析已完成 - 特征: {', '.join(selected_features)}",
                "用户强制中断了元数据分析任务"
            )
            
            # 设置按钮为运行状态
            self.run_analyzer_button.setText("正在运行...")
            self.run_analyzer_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动元数据分析进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_analyzer_button.setText("开始分析")
            self.run_analyzer_button.setEnabled(True)
    def on_run_filter_clicked(self):
        """Filter 运行按钮点击事件"""
        # 获取输入路径
        input_path = self.filter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 获取过滤模式
        filter_mode = self.filter_mode_combobox.currentText()
        
        # 构建命令
        filter_script = project_root / "FacesetProcessor" / "Filter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(filter_script),
            "--input", str(input_path_obj),
            "--mode", filter_mode,
        ]
        
        # 根据模式添加参数
        if filter_mode == "position":
            # 位置过滤模式：添加 eps 和 min-samples 参数
            try:
                eps = float(self.filter_position_eps_input.text().strip() or "50")
                min_samples = int(self.filter_position_min_samples_input.text().strip() or "2")
                cmd.extend(["--threshold", str(eps), "--min-samples", str(min_samples)])
            except ValueError:
                print("错误: 请输入有效的数字参数")
                return
        elif filter_mode == "faceid":
            # 人脸ID过滤模式：添加 eps, min-samples 参数
            try:
                eps = float(self.filter_faceid_eps_input.text().strip() or "0.3")
                min_samples = int(self.filter_faceid_min_samples_input.text().strip() or "2")
                cmd.extend([
                    "--eps", str(eps),
                    "--min-samples", str(min_samples)
                ])
            except ValueError:
                print("错误: 请输入有效的数字参数")
                return
        elif filter_mode == "repeated":
            # 重复去重模式：添加 threshold 参数（固定使用phash）
            try:
                threshold = float(self.filter_repeated_threshold_input.text().strip() or "0.98")
                cmd.extend([
                    "--threshold", str(threshold)
                ])
            except ValueError:
                print("错误: 请输入有效的阈值参数")
                return
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        mode_desc = {
            "blur": "模糊过滤",
            "faceid": "人脸ID分组",
            "position": "位置聚类",
            "repeated": "重复去重"
        }
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"{mode_desc.get(filter_mode, '过滤')} - {filter_mode} 模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_filter_button,
                f"人脸集过滤已完成 - 模式: {filter_mode}",
                "用户强制中断了人脸集过滤任务"
            )
            
            # 设置按钮为运行状态
            self.run_filter_button.setText("正在运行...")
            self.run_filter_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸集过滤进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_filter_button.setText("开始过滤")
            self.run_filter_button.setEnabled(True)
    def on_filter_mode_changed(self, mode: str):
        """过滤模式切换事件"""
        if mode == "position":
            # 显示位置参数
            self.filter_position_eps_input.setVisible(True)
            self.filter_position_min_samples_input.setVisible(True)
            # 隐藏人脸ID参数
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
            # 隐藏重复去重参数
            self.filter_repeated_threshold_input.setVisible(False)
        elif mode == "faceid":
            # 显示人脸ID参数
            self.filter_faceid_eps_input.setVisible(True)
            self.filter_faceid_min_samples_input.setVisible(True)
            # 隐藏位置参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            # 隐藏重复去重参数
            self.filter_repeated_threshold_input.setVisible(False)
        elif mode == "repeated":
            # 显示重复去重参数
            self.filter_repeated_threshold_input.setVisible(True)
            # 隐藏位置参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            # 隐藏人脸ID参数
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
        else:
            # 隐藏所有参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
            self.filter_repeated_threshold_input.setVisible(False)
        
        # 动态调整卡片大小 - 只调整高度，保持宽度不变（带动画）
        if hasattr(self, 'filter_mode_combobox'):
            # 找到包含过滤模式的卡片
            card = self.filter_mode_combobox.parent()
            while card and not hasattr(card, 'setTitle'):
                card = card.parent()
            if card:
                # 保存当前宽度和高度
                current_width = card.width()
                current_height = card.height()
                
                # 临时取消固定宽度限制，让 adjustSize 能正确计算
                card.setMinimumWidth(0)
                card.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
                
                # 强制刷新布局
                card.layout().invalidate()
                card.layout().activate()
                QApplication.processEvents()
                
                # 使用定时器延迟计算目标高度，确保布局已完全更新
                QTimer.singleShot(10, lambda: self._calculate_and_animate(card, current_width, current_height))
    
    def _calculate_and_animate(self, card, current_width, current_height):
        """延迟计算目标高度并启动动画"""
        # 计算目标高度
        card.adjustSize()
        target_height = card.sizeHint().height()
        
        # 恢复宽度限制并设置当前宽度
        card.setFixedWidth(current_width)
        
        # 如果目标高度与当前高度相同，不需要动画
        if abs(target_height - current_height) < 1:
            return
        
        # 创建或获取高度动画
        if id(card) not in self.card_height_animations:
            animation = SiExpAnimation(card)
            animation.setFactor(1/8)
            animation.setBias(0.2)
            animation.ticked.connect(lambda value, c=card: c.setFixedHeight(int(value)))
            self.card_height_animations[id(card)] = animation
        
        # 设置动画的起始和目标值
        animation = self.card_height_animations[id(card)]
        animation.setCurrent(current_height)
        animation.setTarget(target_height)
        animation.start()
    def on_merge_subfolders_clicked(self):
        """合并子文件夹按钮点击事件"""
        # 获取输入路径
        input_path = self.filter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 安全检查：确保不是系统根目录
        dangerous_paths = ['C:\\', 'D:\\', 'E:\\', 'F:\\', '/', '/home', '/usr']
        if str(input_path_obj) in dangerous_paths or len(str(input_path_obj)) <= 3:
            print(f"错误: 出于安全考虑，不允许在根目录执行此操作: {input_path_obj}")
            return
        
        # 构建命令
        filter_script = project_root / "FacesetProcessor" / "Filter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(filter_script),
            "--input", str(input_path_obj),
            "--merge-only",
            "--output-dir", "aligned"
        ]
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print("提示: 只遍历一级子文件夹，不会递归到更深层")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                "合并子文件夹 - 安全模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.merge_subfolders_button,
                "子文件夹合并已完成",
                "用户强制中断了子文件夹合并任务"
            )
            
            # 设置按钮为运行状态
            self.merge_subfolders_button.setText("正在合并...")
            self.merge_subfolders_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动子文件夹合并进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.merge_subfolders_button.setText("合并子文件夹")
            self.merge_subfolders_button.setEnabled(True)
    def on_run_sorter_clicked(self):
        """Sorter 运行按钮点击事件"""
        # 获取输入路径
        input_path = self.sorter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 获取排序方式
        sort_method = self.sorter_method_combobox.currentText()
        
        # 构建命令
        sorter_script = project_root / "FacesetProcessor" / "Sorter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(sorter_script),
            "--input", str(input_path_obj),
            "--method", sort_method,
        ]
        
        # 根据排序方法添加额外参数
        description_parts = [f"方法: {sort_method}"]
        
        if sort_method == "face_pose":
            # 添加姿态类型参数
            pose_type = self.sorter_pose_type_combobox.currentText()
            cmd.extend(["--pose-type", pose_type])
            description_parts.append(f"姿态: {pose_type}")
        elif sort_method == "blur":
            # 添加运动模糊参数
            if self.sorter_motion_blur_checkbox.isChecked():
                cmd.append("--motion-blur")
                description_parts.append("运动模糊")
        
        # 始终重命名文件（排序的核心功能）
        cmd.extend(["--rename", "--prefix", "sorted"])
        description_parts.append("自动重命名+更新元数据")
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"人脸集排序 - {', '.join(description_parts)}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_sorter_button,
                f"人脸集排序已完成 - {', '.join(description_parts)}",
                "用户强制中断了人脸集排序任务"
            )
            
            # 设置按钮为运行状态
            self.run_sorter_button.setText("正在运行...")
            self.run_sorter_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸集排序进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_sorter_button.setText("开始排序")
            self.run_sorter_button.setEnabled(True)
