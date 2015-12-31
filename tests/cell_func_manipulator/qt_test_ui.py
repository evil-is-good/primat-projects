# -*- coding: utf-8 -*-
import struct
import binascii
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from PyQt4 import QtGui
from PyQt4 import QtCore
# from PyQt4 import QtUiTools
from PyQt4.QtGui import QApplication, QMainWindow
# from test_ui import Ui_MainWindow
import test_ui

class ExampleApp(QMainWindow, test_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        print self.__class__
        self.setupUi(self)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()

# class LoL(Ui_MainWindow):
#     def __init__(self,parent):
#         Ui_MainWindow.__init__(self,parent)
        # self._connectSlots()
# class MyWidget(QMainWindow):
#     def __init__(self, *args):  
#         apply(QMainWindow.__init__, (self,) + args)
#
#         loader = QtUiTools.QUiLoader()
#         file = QtCore.QFile("t2.ui")
#         file.open(QtCore.QFile.ReadOnly)
#         self.myWidget = loader.load(file, self)
#         # self.myWidget.setupUi()
#         # self.textField = QtGui.QLineEdit(self)
#         # self.textField.setText('xx')
#         # self.layout.addWidget(self.textField, 0, 0)
#         file.close()
#
#         # self.setCentralWidget(self.myWidget)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MyWidget()
#     # QtCore.QObject.connect(window.butQuit, QtCore.SIGNAL("clicked()"), QtGui.qApp.quit)
#     app.connect(app, QtCore.SIGNAL("lastWindowClosed()"),
#                 app, QtCore.SLOT("quit()"))
#     window.show()
#     # frame = LoL(None)
#     # frame.show()
#     sys.exit( app.exec_() )
