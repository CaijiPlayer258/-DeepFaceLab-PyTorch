import sys
import os
import ctypes

_DEBUG = '--debug' in sys.argv or os.environ.get('UI_DEBUG', '') == '1'

def _dbg(msg):
    if _DEBUG:
        print(f'Debug: {msg}')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
qt_bin_path = os.path.join(project_root, 'python', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'bin')

if os.path.exists(qt_bin_path):
    _dbg(f'Qt bin path exists: {qt_bin_path}')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(qt_bin_path)
        _dbg('Added DLL directory')
    os.environ['PATH'] = qt_bin_path + ';' + os.environ['PATH']
    _dbg('Updated PATH')

    # Preload critical VC++ + Qt DLLs (only Qt5Core is mandatory)
    vc_dlls = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'concrt140.dll']
    for dll_name in vc_dlls:
        dll_path = os.path.join(qt_bin_path, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
                _dbg(f'Loaded {dll_name}')
            except Exception as e:
                if _DEBUG:
                    _dbg(f'Failed to load {dll_name}: {e}')
    # Qt5Core is mandatory
    qt5core_path = os.path.join(qt_bin_path, 'Qt5Core.dll')
    if os.path.exists(qt5core_path):
        try:
            ctypes.CDLL(qt5core_path)
        except Exception as e:
            raise RuntimeError(f'Failed to load Qt5Core.dll: {e}') from e
    # Other Qt DLLs load lazily — skip preloading
    for dll_name in ['Qt5Gui.dll', 'Qt5Widgets.dll', 'Qt5Svg.dll', 'Qt5Network.dll']:
        dll_path = os.path.join(qt_bin_path, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
            except Exception:
                pass  # will be loaded on demand by Qt5Core

    # Torch lib path — don't preload torch DLLs (heavy, fail-safe lazy)
    torch_lib_path = os.path.join(project_root, 'python', 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        os.environ['PATH'] = torch_lib_path + ';' + os.environ['PATH']
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib_path)
else:
    raise RuntimeError(f'Qt bin path not found: {qt_bin_path}')

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import qInstallMessageHandler
def _silent_qt_msg_handler(msg_type, context, message):
    if 'QPixmap::scaled' in message or 'QPixmap::fromImage' in message:
        return
    if msg_type == 3 or msg_type == 4:
        sys.stderr.write(message + '\n')
qInstallMessageHandler(_silent_qt_msg_handler)

try:
    from ui.ui import MySiliconApp
except ModuleNotFoundError:
    from ui import MySiliconApp

import siui
from siui.core import SiGlobal


def show_version_message(window):
    try:
        icon = SiGlobal.siui.iconpack.get("ic_fluent_hand_wave_filled")
    except KeyError:
        icon = None
    window.LayerRightMessageSidebar().send(
        title="Welcome to 深变:DFL-PyTorch",
        text=" 欢迎使用 深变:DFL-PyTorch",
        msg_type=1,
        icon=icon,
        fold_after=5000,
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    icon_path = os.path.join(script_dir, 'img', 'logo_new.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        if sys.platform == 'win32':
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('DeepFaceLab.Torch.App')
            except Exception:
                pass

    window = MySiliconApp()
    window.show()
    QTimer.singleShot(500, lambda: show_version_message(window))
    sys.exit(app.exec_())
