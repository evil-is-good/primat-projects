# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    label = QtGui.QLabel(u"Привет Даниил!")
    label.show()
    sys.exit(app.exec_())
