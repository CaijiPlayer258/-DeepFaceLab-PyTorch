#!/usr/bin/env python3
"""
批量替换 ui 目录下的 PyQt5 导入为兼容性层导入
"""
import os
import re
from pathlib import Path

def replace_pyqt5_imports(file_path):
    """替换单个文件中的 PyQt5 导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 检查是否包含 PyQt5 导入
    if 'from PyQt5' not in content and 'import PyQt5' not in content:
        return False
    
    # 添加兼容性层导入（如果还没有的话）
    needs_compat_import = 'from core.qtex.qt_compat import' not in content
    
    # 提取所有 PyQt5 导入的模块
    pyqt5_imports = set()
    
    # 匹配 from PyQt5.X import Y, Z
    pattern1 = r'from PyQt5\.(Qt\w+)\s+import\s+([^\n]+)'
    for match in re.finditer(pattern1, content):
        module = match.group(1)
        imports = [imp.strip() for imp in match.group(2).split(',')]
        for imp in imports:
            # 移除可能的 as 别名
            imp_name = imp.split(' as ')[0].strip()
            if imp_name:
                pyqt5_imports.add(imp_name)
    
    if not pyqt5_imports:
        return False
    
    # 删除所有 PyQt5 导入行
    content = re.sub(r'from PyQt5\.Qt\w+\s+import\s+[^\n]+\n?', '', content)
    content = re.sub(r'\n\n+', '\n', content)  # 清理多余空行
    
    # 在文件开头添加兼容性层导入
    if needs_compat_import and pyqt5_imports:
        # 找到第一个 import 语句的位置
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i
                break
        
        compat_import = f"from core.qtex.qt_compat import {', '.join(sorted(pyqt5_imports))}"
        lines.insert(insert_pos, compat_import)
        content = '\n'.join(lines)
    
    # 写入修改后的内容
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Modified: {file_path}")
        return True
    
    return False

def main():
    ui_dir = Path(__file__).parent / 'ui'
    
    modified_count = 0
    for py_file in ui_dir.rglob('*.py'):
        if replace_pyqt5_imports(py_file):
            modified_count += 1
    
    print(f"\nTotal files modified: {modified_count}")

if __name__ == '__main__':
    main()
