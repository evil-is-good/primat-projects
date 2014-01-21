# -*- coding: utf-8 -*-
import sys
from mw import *


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main)
    # QtCore.QObject.connect(main.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), main.close)

    # ui.retranslateUi(main)
    main.show()
    sys.exit(app.exec_())
