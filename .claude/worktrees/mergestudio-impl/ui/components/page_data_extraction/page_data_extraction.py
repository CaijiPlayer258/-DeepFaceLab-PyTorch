from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QBoxLayout
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.container import SiDenseContainer, SiTriSectionPanelCard
from siui.components.editbox import SiLabeledLineEdit
from siui.components.page import SiPage
from siui.components.button import SiPushButtonRefactor
from siui.components import SiTitledWidgetGroup
from siui.components.widgets import SiCheckBox
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
@contextmanager
def createDenseContainer(parent,
                         direction: QBoxLayout.Direction,
                         side: Qt.Edges = Qt.LeftEdge | Qt.TopEdge):
    """创建密集容器的上下文管理器"""
    from siui.components.container import SiDenseContainer
    container = SiDenseContainer(parent)
    container.layout().setDirection(direction)
    container.layout().setSpacing(12)
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    try:
        yield container
    finally:
        parent.addWidget(container, side)
class DataExtractionPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("数据提取")
        # 创建滚动容器
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)
        # 第一组：帧提取相关
        with self.titled_widgets_group as group:
            group.addTitle("帧提取")
            # 原始帧提取卡片
            with createPanelCard(group, "原始帧提取") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 视频路径输入框
                    self.video_path_input = SiLabeledLineEdit(self)
                    self.video_path_input.setTitle("视频路径")
                    self.video_path_input.setPlaceholderText("请输入视频文件路径...")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.video_path_input.setText(str(project_root / "workspace" / "data_dst.mp4"))
                    self.video_path_input.resize(700, 64)
                    # 输出路径输入框
                    self.output_path_input = SiLabeledLineEdit(self)
                    self.output_path_input.setTitle("输出路径")
                    self.output_path_input.setPlaceholderText("请输入输出文件夹路径...")
                    self.output_path_input.setText(str(project_root / "workspace" / "data_dst"))
                    self.output_path_input.resize(700, 64)
                    container.addWidget(self.video_path_input)
                    container.addWidget(self.output_path_input)
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as row2:
                    # FFmpeg 路径输入框
                    self.ffmpeg_path_input = SiLabeledLineEdit(self)
                    self.ffmpeg_path_input.setTitle("FFmpeg 目录")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.ffmpeg_path_input.setText(str(project_root / "ffmpeg"))
                    self.ffmpeg_path_input.setPlaceholderText("FFmpeg 可执行文件所在目录...")
                    self.ffmpeg_path_input.resize(700, 64)
                    self.ffmpeg_path_input.setToolTip("FFmpeg 目录路径，需包含 ffmpeg.exe")
                    row2.addWidget(self.ffmpeg_path_input)
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as row3:
                    # 帧数选择输入框
                    self.frame_count_input = SiLabeledLineEdit(self)
                    self.frame_count_input.setTitle("提取帧数")
                    self.frame_count_input.setPlaceholderText("0=全部")
                    self.frame_count_input.setText("0")
                    self.frame_count_input.resize(150, 48)
                    self.frame_count_input.setToolTip("0=提取全部帧，输入数字提取指定帧数")
                    # 提取按钮
                    self.extract_frames_button = SiPushButtonRefactor(self)
                    self.extract_frames_button.setText("提取帧")
                    self.extract_frames_button.resize(150, 48)
                    self.extract_frames_button.setToolTip("使用 FFmpeg 从视频中提取帧图片")
                    self.extract_frames_button.clicked.connect(self.on_extract_frames_clicked)
                    row3.addWidget(self.frame_count_input)
                    row3.addWidget(self.extract_frames_button)
            # 导出格式选择卡片
            with createPanelCard(group, "导出设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container:
                    # 图片格式选择组合框
                    self.image_format_combobox = SiCapsuleComboBox(self)
                    self.image_format_combobox.setTitle("图片格式")
                    self.image_format_combobox.setMinimumHeight(36)
                    self.image_format_combobox.setEditable(False)
                    self.image_format_combobox.addItems(["png", "jpg"])
                    self.image_format_combobox.setToolTip("选择导出图片的格式")
                    # 编码方式选择组合框
                    self.hardware_acceleration_combobox = SiCapsuleComboBox(self)
                    self.hardware_acceleration_combobox.setTitle("编码方式")
                    self.hardware_acceleration_combobox.setMinimumHeight(36)
                    self.hardware_acceleration_combobox.setEditable(False)
                    self.hardware_acceleration_combobox.addItems([
                        "h264_nvenc",
                        "hevc_nvenc",
                        "h264_amf",
                        "hevc_amf",
                        "h264_qsv",
                        "hevc_qsv",
                        "libx264",
                        "libx265",
                        "copy",
                    ])
                    self.hardware_acceleration_combobox.setToolTip("选择 FFmpeg 视频编码方式")
                    # 处理模式选择组合框
                    self.process_mode_combobox = SiCapsuleComboBox(self)
                    self.process_mode_combobox.setTitle("处理模式")
                    self.process_mode_combobox.setMinimumHeight(36)
                    self.process_mode_combobox.setEditable(False)
                    self.process_mode_combobox.addItems(["处理单个文件", "处理目录"])
                    self.process_mode_combobox.setToolTip("选择处理方式：单个文件或整个目录")
                    container.addWidget(self.image_format_combobox)
                    container.addWidget(self.hardware_acceleration_combobox)
                    container.addWidget(self.process_mode_combobox)
        # 第二组：人脸提取相关
        with self.titled_widgets_group as group:
            group.addTitle("人脸提取")
            # 人脸提取路径卡片
            with createPanelCard(group, "人脸提取路径") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.face_input_path_input = SiLabeledLineEdit(self)
                    self.face_input_path_input.setTitle("输入路径")
                    self.face_input_path_input.setPlaceholderText("请输入视频文件或文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst"
                    self.face_input_path_input.setText(str(default_input))
                    self.face_input_path_input.resize(700, 64)
                    # 输出路径输入框
                    self.face_output_path_input = SiLabeledLineEdit(self)
                    self.face_output_path_input.setTitle("输出路径")
                    self.face_output_path_input.setPlaceholderText("请输入输出文件夹路径...")
                    # 使用相对于项目根目录的路径
                    default_output = project_root / "workspace" / "data_dst" / "aligned"
                    self.face_output_path_input.setText(str(default_output))
                    self.face_output_path_input.resize(700, 64)
                    container.addWidget(self.face_input_path_input)
                    container.addWidget(self.face_output_path_input)
            # 人脸提取运行设置卡片
            with createPanelCard(group, "运行设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：人脸检测算法 + 特征点标记算法 + 输出格式
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        # 人脸检测算法选择组合框（与Extractor.py保持一致）
                        self.face_detector_combobox = SiCapsuleComboBox(self)
                        self.face_detector_combobox.setTitle("人脸检测算法")
                        self.face_detector_combobox.setMinimumHeight(36)
                        self.face_detector_combobox.setEditable(False)
                        self.face_detector_combobox.setMaxVisibleItems(5)
                        self.face_detector_combobox.addItems([
                            "BlazeFace",
                            "CenterFace",
                            "S3FD",
                            "YoloV5Face",
                            "YoloV8Face"
                        ])
                        self.face_detector_combobox.setCurrentText("YoloV8Face")
                        self.face_detector_combobox.setToolTip("选择人脸检测算法\n推荐: YoloV8Face (最快最准)")
                        # 特征点标记算法选择组合框（与Extractor.py保持一致）
                        self.landmark_detector_combobox = SiCapsuleComboBox(self)
                        self.landmark_detector_combobox.setTitle("特征点标记算法")
                        self.landmark_detector_combobox.setMinimumHeight(36)
                        self.landmark_detector_combobox.setEditable(False)
                        self.landmark_detector_combobox.addItems([
                            "insightface-2d106det",
                            "2DFAN-4",
                            "Google-mediapipe"
                        ])
                        self.landmark_detector_combobox.setCurrentText("insightface-2d106det")
                        self.landmark_detector_combobox.setToolTip("选择特征点标记算法")
                        # 输出格式选择组合框
                        self.face_output_format_combobox = SiCapsuleComboBox(self)
                        self.face_output_format_combobox.setTitle("输出格式")
                        self.face_output_format_combobox.setMinimumHeight(36)
                        self.face_output_format_combobox.setEditable(False)
                        self.face_output_format_combobox.addItems(["jpg", "png"])
                        self.face_output_format_combobox.setCurrentText("jpg")
                        self.face_output_format_combobox.setToolTip("选择人脸图片输出格式")
                        row1.addWidget(self.face_detector_combobox)
                        row1.addWidget(self.landmark_detector_combobox)
                        row1.addWidget(self.face_output_format_combobox)
                    # 第三行：脸型类型 + 处理模式 + 预缩放尺寸 + 检测角度
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row3:
                        # 脸型类型选择组合框（对应原版DeepFaceLab的face_type）
                        self.face_type_combobox = SiCapsuleComboBox(self)
                        self.face_type_combobox.setTitle("脸型类型")
                        self.face_type_combobox.setMinimumHeight(36)
                        self.face_type_combobox.setEditable(False)
                        self.face_type_combobox.addItems([
                            "half_face",
                            "midfull_face",
                            "full_face",
                            "whole_face",
                            "head"
                        ])
                        self.face_type_combobox.setCurrentText("whole_face")
                        self.face_type_combobox.setToolTip(
                            "选择提取的脸型范围:\n"
                            "- half_face: 半脸\n"
                            "- midfull_face: 中全脸\n"
                            "- full_face: 全脸\n"
                            "- whole_face: 整脸（包含更多头部区域）\n"
                            "- head: 头部（包含整个头部）"
                        )
                        # 处理模式选择组合框
                        self.process_mode_combobox = SiCapsuleComboBox(self)
                        self.process_mode_combobox.setTitle("处理模式")
                        self.process_mode_combobox.setMinimumHeight(36)
                        self.process_mode_combobox.setEditable(False)
                        self.process_mode_combobox.addItems(["单文件/图片", "视频目录批量"])
                        self.process_mode_combobox.setCurrentText("单文件/图片")
                        self.process_mode_combobox.setToolTip(
                            "选择处理模式:\n"
                            "- 单文件/图片: 处理单个视频文件或图片目录\n"
                            "- 视频目录批量: 遍历目录下所有视频，统一输出到同一文件夹"
                        )
                        # 预缩放尺寸输入框
                        self.resize_input = SiLabeledLineEdit(self)
                        self.resize_input.setTitle("预缩放尺寸")
                        self.resize_input.setPlaceholderText("0=禁用，推荐720")
                        self.resize_input.setText("720")
                        self.resize_input.resize(200, 48)
                        self.resize_input.setToolTip("输入图像预缩放宽度（像素），0表示禁用。当原始分辨率小于此值时自动禁用")
                        
                        # 检测角度输入框
                        self.detection_angles_input = SiLabeledLineEdit(self)
                        self.detection_angles_input.setTitle("检测角度")
                        self.detection_angles_input.setPlaceholderText("0,90,180,270")
                        self.detection_angles_input.setText("0")
                        self.detection_angles_input.resize(250, 48)
                        self.detection_angles_input.setToolTip(
                            "人脸检测角度（逗号分隔），例如:\n"
                            "- 0: 仅正常方向（最快）\n"
                            "- 0,90,180,270: 多方向检测（适合侧脸、倒置等场景）\n"
                            "默认: 0"
                        )
                        row3.addWidget(self.face_type_combobox)
                        row3.addWidget(self.process_mode_combobox)
                        row3.addWidget(self.resize_input)
                        row3.addWidget(self.detection_angles_input)
                    # 第四行：运行按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row4:
                        # 运行按钮
                        self.run_extractor_button = SiPushButtonRefactor(self)
                        self.run_extractor_button.setText("开始提取人脸")
                        self.run_extractor_button.resize(200, 48)
                        self.run_extractor_button.setToolTip("点击开始执行人脸提取任务")
                        self.run_extractor_button.clicked.connect(self.on_run_extractor_clicked)
                        
                        row4.addWidget(self.run_extractor_button)
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
    def on_extract_frames_clicked(self):
        """FFmpeg 提取视频帧"""
        video_path = self.video_path_input.text().strip()
        if not video_path:
            print("错误: 请输入视频路径")
            return

        output_path = self.output_path_input.text().strip()
        if not output_path:
            print("错误: 请输入输出路径")
            return

        project_root = Path(__file__).parent.parent.parent.parent
        video_path_obj = Path(video_path)
        if not video_path_obj.is_absolute():
            video_path_obj = project_root / video_path
        if not video_path_obj.exists():
            print(f"错误: 视频文件不存在: {video_path_obj}")
            return

        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = project_root / output_path
        output_path_obj.mkdir(parents=True, exist_ok=True)

        # 帧数
        frame_count_str = self.frame_count_input.text().strip()
        try:
            frame_count = int(frame_count_str) if frame_count_str else 0
        except ValueError:
            frame_count = 0

        # 图片格式
        img_fmt = self.image_format_combobox.currentText()

        # FFmpeg 路径（可自定义）
        ffmpeg_dir_str = self.ffmpeg_path_input.text().strip()
        ffmpeg_dir = Path(ffmpeg_dir_str) if ffmpeg_dir_str else (project_root / "ffmpeg")
        if not ffmpeg_dir.is_absolute():
            ffmpeg_dir = project_root / ffmpeg_dir
        ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
        if not ffmpeg_exe.exists():
            print(f"错误: FFmpeg 未找到: {ffmpeg_exe}")
            return

        # 构建命令
        output_pattern = output_path_obj / f"%06d.{img_fmt}"
        cmd = [
            str(ffmpeg_exe),
            "-i", str(video_path_obj),
            "-q:v", "2",
        ]
        if frame_count > 0:
            # -vframes 限制帧数
            cmd.extend(["-vframes", str(frame_count)])
        cmd.append(str(output_pattern))

        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")

        parent_window = self.window()
        count_info = f"全部帧" if frame_count <= 0 else f"{frame_count} 帧"
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(cmd_str, f"FFmpeg 提取帧 - {count_info} → {img_fmt}")

        try:
            monitor = self._create_task_monitor(
                self.extract_frames_button,
                f"帧提取完成 - {count_info}",
                "用户中断了帧提取任务"
            )
            self.extract_frames_button.setText("正在提取...")
            self.extract_frames_button.setEnabled(False)

            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                threading.Thread(target=lambda p=process: monitor(p), daemon=True).start()
            else:
                process = subprocess.Popen(cmd)
                threading.Thread(target=lambda p=process: monitor(p), daemon=True).start()
            print("✓ 已启动 FFmpeg 帧提取进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.extract_frames_button.setText("提取帧")
            self.extract_frames_button.setEnabled(True)

    def on_run_extractor_clicked(self):
        """运行按钮点击事件"""
        # 获取输入路径
        input_path = self.face_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的输入路径")
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
        
        # 获取输出路径
        output_path = self.face_output_path_input.text().strip()
        if not output_path:
            print("错误: 请输入有效的输出路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = project_root / output_path
        
        # 获取检测器和标记器
        detector = self.face_detector_combobox.currentText()
        landmarker = self.landmark_detector_combobox.currentText()
        
        # 获取脸型类型
        face_type = self.face_type_combobox.currentText()
        
        # 获取预缩放参数
        resize_value_str = self.resize_input.text().strip()
        try:
            resize_value = int(resize_value_str) if resize_value_str else 0
            if resize_value < 0:
                resize_value = 0
        except ValueError:
            print("警告: 无效的预缩放尺寸，已重置为0（禁用）")
            resize_value = 0
        
        # 获取处理模式
        process_mode = self.process_mode_combobox.currentText()
        is_video_batch = (process_mode == "视频目录批量")
        
        # 获取检测角度
        detection_angles_str = self.detection_angles_input.text().strip()
        if not detection_angles_str:
            detection_angles_str = "0"  # 默认为0度
        
        # 构建命令（使用绝对路径）
        extractor_script = project_root / "Extractor" / "Extractor.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(extractor_script),
            "-i", str(input_path_obj),
            "-o", str(output_path_obj),
            "-d", detector,
            "-l", landmarker,
            "-t", face_type,
            "-a", detection_angles_str  # 添加检测角度参数
            # 注意：UI模式强制禁用快速测试（debug）模式
        ]
        
        # 添加预缩放参数（仅在非0时添加）
        if resize_value > 0:
            cmd.extend(["-r", str(resize_value)])
        
        # 添加视频目录批量模式参数
        if is_video_batch:
            cmd.extend(["-m", "video"])
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        # 显示通知
        parent_window = self.window()
        resize_info = f" (预缩放: {resize_value}px)" if resize_value > 0 else " (无预缩放)"
        batch_mode_info = " [视频批量模式]" if is_video_batch else ""
        angles_info = f", 角度: {detection_angles_str}" if detection_angles_str != "0" else ""
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"人脸提取 - {detector} + {landmarker}, 脸型: {face_type}{angles_info}{resize_info}{batch_mode_info}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_extractor_button,
                f"人脸提取已完成 - {detector} + {landmarker}, 脸型: {face_type}{angles_info}{resize_info}{batch_mode_info}",
                "用户强制中断了人脸提取任务"
            )
            
            # 设置按钮为运行状态
            self.run_extractor_button.setText("正在运行...")
            self.run_extractor_button.setEnabled(False)
            
            # 在新窗口中运行命令，并在执行完成后等待用户按键
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                # Windows 系统：使用 cmd /c 执行命令，完成后 pause 等待按键
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                # 其他系统
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸提取进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_extractor_button.setText("开始提取人脸")
            self.run_extractor_button.setEnabled(True)
