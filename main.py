# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from ui import Ui_Widget
import AE
import LSTM
import Z


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # button handler
        self.ui.pushButton.clicked.connect(self.start_detection)

        # initialize timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_z_score)

    def start_detection(self):
        """initialize AE, LSTM Z-score detection"""
        # run AE LSTM algorithm and show results
        self.run_once_algorithms()

        # initialize Z-score realtime detection
        self.timer.start(500)  # 500ms refresh rate for Z-score

    def update_z_score(self):
        """update Z-score results"""
        Z.run_z_score_algorithm()
        self.display_image('z_score_output.png', self.ui.graphicsViewZ)

    def run_once_algorithms(self):
        """run AE and LSTM once and show result"""
        AE.run_ae_algorithm()
        self.display_image('ae_output.png', self.ui.graphicsViewAE)

        LSTM.run_lstm_algorithm()
        self.display_image('lstm_output.png', self.ui.graphicsViewLSTM)

    def display_image(self, image_path, graphics_view):
        """show in QGraphicsView """
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphics_view.setScene(scene)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
