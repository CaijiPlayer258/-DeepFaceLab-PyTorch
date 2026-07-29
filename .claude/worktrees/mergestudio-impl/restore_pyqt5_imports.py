#!/usr/bin/env python3
"""
恢复 ui 目录下的兼容性层导入为 PyQt5 导入
"""
import re
from pathlib import Path

def restore_pyqt5_imports(file_path):
    """恢复单个文件中的 PyQt5 导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    compat_imports = {}
    
    for line in lines:
        # 检查是否是兼容性层导入
        if 'from core.qtex.qt_compat import' in line:
            # 提取所有导入的符号
            match = re.search(r'from core\.qtex\.qt_compat import (.+)', line)
            if match:
                imports = [imp.strip() for imp in match.group(1).split(',')]
                # 按模块分类
                for imp in imports:
                    imp_name = imp.split(' as ')[0].strip()
                    # 简单分类（这可能需要更复杂的逻辑）
                    if imp_name in ['QTimer', 'QObject', 'pyqtSignal', 'QIcon', 'QDesktopWidget', 
                                   'QVBoxLayout', 'QWidget', 'QApplication', 'Qt', 'QUrl', 
                                   'QSizePolicy', 'QDialog', 'QRadioButton', 'QDialogButtonBox',
                                   'QLineEdit', 'QPoint', 'QRectF', 'QThread', 'QPolygonF',
                                   'QPainter', 'QPen', 'QBrush', 'QColor', 'QImage', 'QPixmap']:
                        compat_imports[imp_name] = True
            modified = True
        else:
            new_lines.append(line)
    
    if modified and compat_imports:
        # 添加 PyQt5 导入
        pyqt_imports = {
            'QtCore': ['QTimer', 'QObject', 'pyqtSignal', 'Qt', 'QUrl', 'QPoint', 'QRectF', 'QThread'],
            'QtGui': ['QIcon', 'QImage', 'QPixmap', 'QPainter', 'QPen', 'QBrush', 'QColor', 'QPolygonF'],
            'QtWidgets': ['QApplication', 'QDesktopWidget', 'QVBoxLayout', 'QWidget', 'QSizePolicy', 
                         'QDialog', 'QRadioButton', 'QDialogButtonBox', 'QLineEdit']
        }
        
        # 找到插入位置
        insert_pos = 0
        for i, line in enumerate(new_lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i
                break
        
        # 插入 PyQt5 导入
        pyqt5_lines = []
        for module, symbols in pyqt_imports.items():
            needed_symbols = [s for s in symbols if s in compat_imports]
            if needed_symbols:
                pyqt5_lines.append(f"from PyQt5.{module} import {', '.join(needed_symbols)}\n")
        
        for i, line in enumerate(pyqt5_lines):
            new_lines.insert(insert_pos + i, line)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Restored: {file_path}")
        return True
    
    return False

def main():
    ui_dir = Path(__file__).parent / 'ui'
    
    restored_count = 0
    for py_file in ui_dir.rglob('*.py'):
        if restore_pyqt5_imports(py_file):
            restored_count += 1
    
    print(f"\nTotal files restored: {restored_count}")

if __name__ == '__main__':
    main()
