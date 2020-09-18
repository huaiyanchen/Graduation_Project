import sys
from PyQt5 import QtCore, QtWidgets, uic
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PyQt5.QtCore import Qt
qtCreatorFile = "face.ui"  # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.image1 = QPixmap()
        self.image2 = QPixmap()
        self.show_()

    def show_(self):
        
        path1 = 'image\\hd_00_00_7500.jpg'
        path2 = 'image\\hd_00_00_7600.jpg'
        
        self.image1.load(path1)
        self.image2.load(path2)
        self.image1 = self.image1.scaled(1000, 500, aspectRatioMode = Qt.KeepAspectRatio)
        self.image2 = self.image2.scaled(1000, 500, aspectRatioMode=Qt.KeepAspectRatio)
        self.img1.scene = QGraphicsScene()
        item = QGraphicsPixmapItem(self.image1)
        self.img1.scene.addItem(item)
        self.img1.setScene(self.img1.scene)

        self.img2.scene = QGraphicsScene()
        item = QGraphicsPixmapItem(self.image2)
        self.img2.scene.addItem(item)
        self.img2.setScene(self.img2.scene)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    app.f.close()