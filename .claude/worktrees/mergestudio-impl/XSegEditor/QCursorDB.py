from core.qtex.qt_compat import *
                                       
class QCursorDB():
    # Do not construct any QPixmap/QCursor at import time.
    # Some Qt bindings abort the process if GUI objects are created
    # before a QGuiApplication/QApplication exists.
    cross_red = None
    cross_green = None
    cross_blue = None

    @staticmethod
    def initialize(cursor_path):
        try:
            QCursorDB.cross_red = QCursor(QPixmap(str(cursor_path / 'cross_red.png')))
            QCursorDB.cross_green = QCursor(QPixmap(str(cursor_path / 'cross_green.png')))
            QCursorDB.cross_blue = QCursor(QPixmap(str(cursor_path / 'cross_blue.png')))
        except Exception:
            # Keep defaults.
            pass
