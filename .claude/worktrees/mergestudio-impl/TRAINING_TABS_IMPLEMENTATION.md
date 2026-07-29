# 训练界面双页面功能实现总结

## 功能概述

已成功为训练界面创建了**两个独立的子页面**，分别用于不同的场景：

1. **TrainingConfigChildPage** - 点击"训练"按钮时弹出（完整配置）
2. **NewModelConfigChildPage** - 点击"新建模型"按钮时弹出（仅模型架构）

## 主要改动

### 1. TrainingConfigChildPage 类（训练配置页面）

**用途**: 当用户点击现有模型的"训练"按钮时弹出

**包含内容**:
- 模型架构参数（模型名称、分辨率、架构、子分支、e_dims、ae_dims、d_dims、d_mask_dims）
- 数据集参数（src/dst人脸集路径、人脸类型）
- 训练参数（设备、进程数、批次大小、精度、目标迭代数等）
- 优化器参数（AdaBelief、学习率、余弦退火）
- 数据增强参数（旋转、缩放、偏移、颜色模式、HSV等）
- GAN训练参数

**特性**:
- 架构选择使用胶囊下拉框（DF、LIAE、AMP）
- 子分支选择使用胶囊下拉框（-u、-ud、-ut、-udt、-d、-dt、-t）
- 动态描述更新：选择不同选项时，卡片副标题实时更新

### 2. NewModelConfigChildPage 类（新建模型配置页面）

**用途**: 当用户点击"新建模型"按钮时弹出

**包含内容**:
- 仅包含模型架构参数（模型名称、分辨率、架构、子分支、e_dims、ae_dims、d_dims、d_mask_dims）
- 不包含训练相关的复杂配置

**特性**:
- 简洁的界面，专注于模型架构选择
- 架构选择使用胶囊下拉框（DF、LIAE、AMP）
- 子分支选择使用胶囊下拉框（-u、-ud、-ut、-udt、-d、-dt、-t）
- 动态描述更新：选择不同选项时，卡片副标题实时更新
- 底部按钮为"创建模型"和"取消"

### 3. 架构选择改进

#### 胶囊下拉框替换
将原来的文本输入框 `SiLabeledLineEdit` 替换为 `SiCapsuleComboBox`：

**架构选项**：
- DF (DeepFake架构)
- LIAE (轻量级交互式自动编码器)
- AMP (自适应多精度训练)

**子分支选项**：
- `-u`: 仅使用编码器
- `-ud`: 编码器和解码器
- `-ut`: 编码器和转换器
- `-udt`: 编码器、解码器和转换器
- `-d`: 仅使用解码器
- `-dt`: 解码器和转换器
- `-t`: 仅使用转换器

### 3. 动态描述更新

当用户选择不同的架构或子分支时，卡片的副标题会实时更新，提供更详细的说明信息。

例如：
- 选择 "DF" → "DeepFake架构 - 经典的单编码器-单解码器结构"
- 选择 "-ud" → "编码器和解码器 (Encoder + Decoder)"

## 技术实现细节

### 页面调用机制

**训练按钮点击事件**:
```python
def open_training_config(self, model_name: str):
    """打开训练配置子页面（点击训练按钮时）"""
    main_window = SiGlobal.siui.windows.get("MAIN_WINDOW")
    if main_window and hasattr(main_window, 'layerChildPage'):
        child_page = TrainingConfigChildPage(self, model_name)
        main_window.layerChildPage().setChildPage(child_page)
```

**新建模型按钮点击事件**:
```python
def open_new_model_config(self, model_name: str):
    """打开新建模型配置子页面（点击新建模型按钮时）"""
    main_window = SiGlobal.siui.windows.get("MAIN_WINDOW")
    if main_window and hasattr(main_window, 'layerChildPage'):
        child_page = NewModelConfigChildPage(self, model_name)
        main_window.layerChildPage().setChildPage(child_page)
```

### 架构选择器
```python
self.archi_combo = SiCapsuleComboBox(self.archi_card)
self.archi_combo.setFixedWidth(200)
self.archi_combo.setMinimumHeight(32)
self.archi_combo.setMaximumHeight(32)
self.archi_combo.setEditable(False)
self.archi_combo.addItems(["DF", "LIAE", "AMP"])
self.archi_combo.currentTextChanged.connect(self.update_architecture)
```

## 用户体验提升

1. **清晰的场景分离**: 
   - 训练时显示完整配置，方便调整所有参数
   - 新建模型时仅显示架构选择，简化操作流程

2. **直观的选项展示**: 
   - 使用胶囊下拉框代替文本输入，提供明确的选项列表
   - 避免用户输入错误的架构名称

3. **实时反馈**: 
   - 选择不同选项时，卡片副标题立即更新，提供详细说明
   - 帮助用户理解每个选项的含义

4. **简洁的界面设计**: 
   - NewModelConfigChildPage 专注于架构选择，减少认知负担
   - TrainingConfigChildPage 提供全面配置，满足高级需求

## 测试验证

运行测试脚本 `test_training_tabs.py` 验证了：
- ✓ TrainingConfigChildPage 成功创建（训练配置页面）
- ✓ NewModelConfigChildPage 成功创建（新建模型配置页面）
- ✓ TrainingConfigChildPage 架构下拉框包含3个选项（DF, LIAE, AMP）
- ✓ TrainingConfigChildPage 子分支下拉框包含7个选项（-u, -ud, -ut, -udt, -d, -dt, -t）
- ✓ NewModelConfigChildPage 架构下拉框包含3个选项
- ✓ NewModelConfigChildPage 子分支下拉框包含7个选项
- ✓ 两个页面的架构和子分支选择器功能一致

## 文件修改

- **修改文件**: `ui/components/page_trainer/page_trainer.py`
- **新增文件**: `test_training_tabs.py` (测试脚本)

## 后续优化建议

1. 可以在 NewModelConfigChildPage 中添加“立即训练”按钮，创建模型后直接进入训练配置
2. 可以保存用户的架构选择偏好，下次新建模型时自动选择上次使用的架构
3. 可以为不同的架构预设不同的默认参数（如 e_dims、ae_dims 等）
4. 可以添加架构对比功能，帮助用户选择合适的架构
5. 可以在 TrainingConfigChildPage 中添加“重置为默认值”按钮
