# DeepFaceLab-Torch 自定义图标配置说明

## 图标文件位置

项目中的自定义图标文件位于：
```
ui/img/
├── logo_new.png        - 主Logo（窗口图标和任务栏图标）
├── empty_icon.png      - 空图标（备用）
├── logo.png            - 旧版Logo
└── ...
```

## 当前使用的图标

### 1. 窗口图标 (Window Title Bar Icon)
- **文件**: `ui/img/logo_new.png`
- **位置**: 应用程序窗口左上角
- **代码位置**: 
  - `ui/start.py` - 应用程序级别设置
  - `ui/ui.py` - 窗口级别设置

### 2. 任务栏图标 (Taskbar Icon)
- **文件**: `ui/img/logo_new.png`
- **位置**: Windows任务栏
- **特殊处理**: 
  - 使用 `QApplication.setWindowIcon()` 设置
  - 设置 Windows AppUserModelID 以确保正确显示

## 如何更换图标

如果你想使用自己的图标，只需替换 `ui/img/logo_new.png` 文件即可。

### 推荐的图标规格

- **格式**: PNG（支持透明背景）
- **尺寸**: 256x256 像素或更大（系统会自动缩放）
- **建议尺寸**: 64x64, 128x128, 256x256（多尺寸适配）
- **透明背景**: 推荐使用透明背景以获得更好的视觉效果

### 更换步骤

1. 准备你的图标文件（PNG格式）
2. 将文件命名为 `logo_new.png`
3. 替换 `ui/img/logo_new.png` 文件
4. 重新启动程序

## 技术实现细节

### 1. start.py 中的设置（推荐方式）

```python
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序图标（任务栏）
    icon_path = os.path.join(script_dir, 'img', 'logo_new.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
        # Windows特定：设置AppUserModelID
        if sys.platform == 'win32':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'DeepFaceLab.Torch.App'
            )
```

### 2. ui.py 中的设置（双重保险）

```python
class MySiliconApp(SiliconApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'img', 
            'logo_new.png'
        )
        if os.path.exists(icon_path):
            window_icon = QIcon(icon_path)
            self.setWindowIcon(window_icon)
            QApplication.setWindowIcon(window_icon)
            
            # Windows特定设置
            if sys.platform == 'win32':
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    'DeepFaceLab.Torch.App'
                )
```

## Windows 任务栏图标注意事项

### 为什么需要 AppUserModelID？

在 Windows 7 及更高版本中，Windows 使用 AppUserModelID 来识别应用程序。如果不设置这个ID，Windows 可能会：
- 不显示自定义图标
- 将多个窗口分组错误
- 使用默认的 Python 图标

### 如果任务栏图标仍然不显示

1. **检查图标文件格式**
   - 确保是有效的 PNG 文件
   - 尝试使用其他图片查看器打开验证

2. **清除 Windows 图标缓存**
   ```powershell
   # 重启 Windows 资源管理器
   taskkill /f /im explorer.exe
   start explorer.exe
   ```

3. **取消固定并重新固定**
   - 右键点击任务栏图标 → "从任务栏取消固定"
   - 重新运行程序
   - 右键点击新图标 → "固定到任务栏"

4. **重启电脑**
   - 有时 Windows 需要重启才能更新图标缓存

## 其他图标的用途

### About 页面 Logo
- **文件**: `ui/img/logo_new.png`
- **位置**: "关于"页面顶部
- **代码**: `ui/components/page_about/page_about.py` 第38行

### 主页背景
- **文件**: `ui/img/homepage_background.png`
- **位置**: 主页顶部背景
- **代码**: `ui/components/page_homepage/page_homepage.py` 第54行

## 故障排除

### 问题1: 图标显示为空白
**解决方案**:
- 检查文件路径是否正确
- 确认 PNG 文件没有损坏
- 查看控制台输出是否有错误信息

### 问题2: 只有标题栏有图标，任务栏没有
**解决方案**:
- 确认已设置 AppUserModelID
- 尝试重启程序
- 清除 Windows 图标缓存

### 问题3: 图标太小或模糊
**解决方案**:
- 使用更高分辨率的 PNG 文件（至少 256x256）
- 确保图标质量良好
- Windows 会自动缩放到合适的大小

## 相关文件清单

- `ui/start.py` - 主启动脚本，设置应用程序图标
- `ui/ui.py` - 主界面类，设置窗口图标
- `ui/img/logo_new.png` - 主图标文件
- `ui/components/page_about/page_about.py` - About页面，使用相同图标

## 测试图标是否正常

运行程序后，检查以下内容：
1. ✓ 窗口左上角显示自定义图标
2. ✓ Windows任务栏显示自定义图标
3. ✓ Alt+Tab 切换窗口时显示自定义图标
4. ✓ "关于"页面显示自定义图标

如果以上都正常，说明图标配置成功！
