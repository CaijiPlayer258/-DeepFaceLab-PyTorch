from core.qtex.qt_compat import *

class QImageDB():
    @staticmethod
    def initialize(image_path):
        QImageDB.intro = QImage ( str(image_path / 'intro.png') )
