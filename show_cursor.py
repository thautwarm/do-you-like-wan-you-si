import sys
import compat

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QCursor

from fgo import interops
try:
    from fgo.model import predict
except:
    def predict():
        return "模型未训练"

class Window(QWidget):
    cursor_move = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        mouse_cursor_board = self.mouse_cursor_board = QLineEdit(self)
        cls_board = self.cls_board = QLineEdit(self)

        mouse_cursor_board.move(50, 120)
        cls_board.move(50, 220)

        timer_for_mouse = QTimer()
        timer_for_mouse.setInterval(500)
        timer_for_mouse.timeout.connect(self.add_mouse_events)
        self.timer_for_mouse = timer_for_mouse

        timer_for_clf = QTimer()
        timer_for_clf.setInterval(1200)
        timer_for_clf.timeout.connect(self.add_clf_events)

        self.timer_for_clf = timer_for_clf

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('board')
        self.setMouseTracking(True)
        self.cursor_move.connect(self.mouse_event)
        timer_for_mouse.start()
        timer_for_clf.start()
        self.show()

    def add_mouse_events(self):
        pos = QCursor.pos()
        if pos != self.cursor:
            self.cursor = pos
            self.cursor_move.emit(pos)

    def add_clf_events(self):
        self.cls_board.setText(predict())

    def mouse_event(self, e):
        x = e.x()
        y = e.y() - 40
        text = "x: {0},  y: {1}".format(x, y)
        self.mouse_cursor_board.setText(text)


if True:
    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())
