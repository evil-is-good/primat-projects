# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cfm.ui'
#
# Created: Fri Nov 13 10:55:18 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(956, 619)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 30, 21, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 10, 16, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 10, 16, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(90, 10, 16, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.kx = QtGui.QLineEdit(self.centralwidget)
        self.kx.setGeometry(QtCore.QRect(40, 30, 21, 27))
        self.kx.setObjectName(_fromUtf8("kx"))
        self.ky = QtGui.QLineEdit(self.centralwidget)
        self.ky.setGeometry(QtCore.QRect(60, 30, 21, 27))
        self.ky.setObjectName(_fromUtf8("ky"))
        self.kz = QtGui.QLineEdit(self.centralwidget)
        self.kz.setGeometry(QtCore.QRect(80, 30, 21, 27))
        self.kz.setObjectName(_fromUtf8("kz"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 70, 21, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.nu = QtGui.QLineEdit(self.centralwidget)
        self.nu.setGeometry(QtCore.QRect(40, 60, 21, 27))
        self.nu.setObjectName(_fromUtf8("nu"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(0, 100, 41, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.orts = QtGui.QLineEdit(self.centralwidget)
        self.orts.setGeometry(QtCore.QRect(40, 90, 31, 27))
        self.orts.setObjectName(_fromUtf8("orts"))
        self.plot_move = QtGui.QPushButton(self.centralwidget)
        self.plot_move.setGeometry(QtCore.QRect(130, 30, 111, 27))
        self.plot_move.setObjectName(_fromUtf8("plot_move"))
        self.plot_deform = QtGui.QPushButton(self.centralwidget)
        self.plot_deform.setGeometry(QtCore.QRect(260, 30, 111, 27))
        self.plot_deform.setObjectName(_fromUtf8("plot_deform"))
        self.plot_stress = QtGui.QPushButton(self.centralwidget)
        self.plot_stress.setGeometry(QtCore.QRect(390, 30, 111, 27))
        self.plot_stress.setObjectName(_fromUtf8("plot_stress"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 130, 141, 17))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.label_9 = QtGui.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 170, 31, 17))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.label_10 = QtGui.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(30, 190, 31, 17))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.label_11 = QtGui.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(30, 210, 31, 17))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.label_12 = QtGui.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(130, 190, 31, 17))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.label_13 = QtGui.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(130, 170, 31, 17))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.label_14 = QtGui.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(130, 210, 31, 17))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.label_15 = QtGui.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(30, 290, 31, 17))
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.label_16 = QtGui.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(30, 270, 31, 17))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.label_17 = QtGui.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(130, 310, 31, 17))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.label_18 = QtGui.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(130, 270, 31, 17))
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.label_19 = QtGui.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(130, 290, 31, 17))
        self.label_19.setObjectName(_fromUtf8("label_19"))
        self.label_20 = QtGui.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(30, 310, 31, 17))
        self.label_20.setObjectName(_fromUtf8("label_20"))
        self.Ex1 = QtGui.QLineEdit(self.centralwidget)
        self.Ex1.setGeometry(QtCore.QRect(70, 160, 51, 27))
        self.Ex1.setObjectName(_fromUtf8("Ex1"))
        self.Ey1 = QtGui.QLineEdit(self.centralwidget)
        self.Ey1.setGeometry(QtCore.QRect(70, 190, 51, 27))
        self.Ey1.setObjectName(_fromUtf8("Ey1"))
        self.Ez1 = QtGui.QLineEdit(self.centralwidget)
        self.Ez1.setGeometry(QtCore.QRect(70, 220, 51, 27))
        self.Ez1.setObjectName(_fromUtf8("Ez1"))
        self.Ey2 = QtGui.QLineEdit(self.centralwidget)
        self.Ey2.setGeometry(QtCore.QRect(170, 190, 51, 27))
        self.Ey2.setObjectName(_fromUtf8("Ey2"))
        self.Ez2 = QtGui.QLineEdit(self.centralwidget)
        self.Ez2.setGeometry(QtCore.QRect(170, 220, 51, 27))
        self.Ez2.setObjectName(_fromUtf8("Ez2"))
        self.Ex2 = QtGui.QLineEdit(self.centralwidget)
        self.Ex2.setGeometry(QtCore.QRect(170, 160, 51, 27))
        self.Ex2.setObjectName(_fromUtf8("Ex2"))
        self.Ny1 = QtGui.QLineEdit(self.centralwidget)
        self.Ny1.setGeometry(QtCore.QRect(70, 290, 51, 27))
        self.Ny1.setObjectName(_fromUtf8("Ny1"))
        self.Ny2 = QtGui.QLineEdit(self.centralwidget)
        self.Ny2.setGeometry(QtCore.QRect(170, 290, 51, 27))
        self.Ny2.setObjectName(_fromUtf8("Ny2"))
        self.Nz1 = QtGui.QLineEdit(self.centralwidget)
        self.Nz1.setGeometry(QtCore.QRect(70, 320, 51, 27))
        self.Nz1.setObjectName(_fromUtf8("Nz1"))
        self.Nx1 = QtGui.QLineEdit(self.centralwidget)
        self.Nx1.setGeometry(QtCore.QRect(70, 260, 51, 27))
        self.Nx1.setObjectName(_fromUtf8("Nx1"))
        self.Nx2 = QtGui.QLineEdit(self.centralwidget)
        self.Nx2.setGeometry(QtCore.QRect(170, 260, 51, 27))
        self.Nx2.setObjectName(_fromUtf8("Nx2"))
        self.Nz2 = QtGui.QLineEdit(self.centralwidget)
        self.Nz2.setGeometry(QtCore.QRect(170, 320, 51, 27))
        self.Nz2.setObjectName(_fromUtf8("Nz2"))
        self.num_mat = QtGui.QLineEdit(self.centralwidget)
        self.num_mat.setGeometry(QtCore.QRect(170, 120, 51, 27))
        self.num_mat.setObjectName(_fromUtf8("num_mat"))
        self.label_8 = QtGui.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(240, 130, 151, 17))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.num_approx = QtGui.QLineEdit(self.centralwidget)
        self.num_approx.setGeometry(QtCore.QRect(400, 120, 51, 27))
        self.num_approx.setObjectName(_fromUtf8("num_approx"))
        self.num_ref = QtGui.QLineEdit(self.centralwidget)
        self.num_ref.setGeometry(QtCore.QRect(400, 160, 51, 27))
        self.num_ref.setObjectName(_fromUtf8("num_ref"))
        self.D = QtGui.QLineEdit(self.centralwidget)
        self.D.setGeometry(QtCore.QRect(400, 200, 51, 27))
        self.D.setObjectName(_fromUtf8("D"))
        self.label_22 = QtGui.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(240, 210, 151, 17))
        self.label_22.setObjectName(_fromUtf8("label_22"))
        self.label_21 = QtGui.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(240, 170, 151, 17))
        self.label_21.setObjectName(_fromUtf8("label_21"))
        self.tableArgs = QtGui.QTableWidget(self.centralwidget)
        self.tableArgs.setGeometry(QtCore.QRect(40, 360, 911, 192))
        self.tableArgs.setShowGrid(True)
        self.tableArgs.setGridStyle(QtCore.Qt.SolidLine)
        self.tableArgs.setRowCount(1)
        self.tableArgs.setColumnCount(18)
        self.tableArgs.setObjectName(_fromUtf8("tableArgs"))
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setVerticalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(4, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(5, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(6, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(7, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(8, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(9, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(10, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(11, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(12, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(13, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(14, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(15, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(16, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setHorizontalHeaderItem(17, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 0, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 1, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 2, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 3, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 4, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 5, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 6, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 7, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 8, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 9, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 10, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 11, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 12, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 13, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 14, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 15, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 16, item)
        item = QtGui.QTableWidgetItem()
        self.tableArgs.setItem(0, 17, item)
        self.tableArgs.horizontalHeader().setDefaultSectionSize(50)
        self.plus = QtGui.QPushButton(self.centralwidget)
        self.plus.setGeometry(QtCore.QRect(10, 360, 31, 27))
        self.plus.setObjectName(_fromUtf8("plus"))
        self.loadTable = QtGui.QPushButton(self.centralwidget)
        self.loadTable.setGeometry(QtCore.QRect(300, 320, 81, 27))
        self.loadTable.setObjectName(_fromUtf8("loadTable"))
        self.tableName = QtGui.QLineEdit(self.centralwidget)
        self.tableName.setGeometry(QtCore.QRect(480, 320, 113, 27))
        self.tableName.setObjectName(_fromUtf8("tableName"))
        self.saveTable = QtGui.QPushButton(self.centralwidget)
        self.saveTable.setGeometry(QtCore.QRect(390, 320, 81, 27))
        self.saveTable.setObjectName(_fromUtf8("saveTable"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "k=", None))
        self.label_2.setText(_translate("MainWindow", "x", None))
        self.label_3.setText(_translate("MainWindow", "y", None))
        self.label_4.setText(_translate("MainWindow", "z", None))
        self.kx.setText(_translate("MainWindow", "1", None))
        self.ky.setText(_translate("MainWindow", "0", None))
        self.kz.setText(_translate("MainWindow", "0", None))
        self.label_5.setText(_translate("MainWindow", "ν=", None))
        self.nu.setText(_translate("MainWindow", "x", None))
        self.label_6.setText(_translate("MainWindow", "orts=", None))
        self.orts.setText(_translate("MainWindow", "xx", None))
        self.plot_move.setText(_translate("MainWindow", "Перемещения", None))
        self.plot_deform.setText(_translate("MainWindow", "Деформации", None))
        self.plot_stress.setText(_translate("MainWindow", "Напряжения", None))
        self.label_7.setText(_translate("MainWindow", "Число материалов", None))
        self.label_9.setText(_translate("MainWindow", "Ex1=", None))
        self.label_10.setText(_translate("MainWindow", "Ey1=", None))
        self.label_11.setText(_translate("MainWindow", "Ez1=", None))
        self.label_12.setText(_translate("MainWindow", "Ey2=", None))
        self.label_13.setText(_translate("MainWindow", "Ex2=", None))
        self.label_14.setText(_translate("MainWindow", "Ez2=", None))
        self.label_15.setText(_translate("MainWindow", "Ny1=", None))
        self.label_16.setText(_translate("MainWindow", "Nx1=", None))
        self.label_17.setText(_translate("MainWindow", "Nz2=", None))
        self.label_18.setText(_translate("MainWindow", "Nx2=", None))
        self.label_19.setText(_translate("MainWindow", "Ny2=", None))
        self.label_20.setText(_translate("MainWindow", "Nz1=", None))
        self.Ex1.setText(_translate("MainWindow", "1.0", None))
        self.Ey1.setText(_translate("MainWindow", "1.0", None))
        self.Ez1.setText(_translate("MainWindow", "1.0", None))
        self.Ey2.setText(_translate("MainWindow", "10.0", None))
        self.Ez2.setText(_translate("MainWindow", "10.0", None))
        self.Ex2.setText(_translate("MainWindow", "10.0", None))
        self.Ny1.setText(_translate("MainWindow", "0.25", None))
        self.Ny2.setText(_translate("MainWindow", "0.25", None))
        self.Nz1.setText(_translate("MainWindow", "0.25", None))
        self.Nx1.setText(_translate("MainWindow", "0.25", None))
        self.Nx2.setText(_translate("MainWindow", "0.25", None))
        self.Nz2.setText(_translate("MainWindow", "0.25", None))
        self.num_mat.setText(_translate("MainWindow", "2", None))
        self.label_8.setText(_translate("MainWindow", "Число приближений", None))
        self.num_approx.setText(_translate("MainWindow", "2", None))
        self.num_ref.setText(_translate("MainWindow", "3", None))
        self.D.setText(_translate("MainWindow", "0.25", None))
        self.label_22.setText(_translate("MainWindow", "Диаметр включения", None))
        self.label_21.setText(_translate("MainWindow", "Число рефайнов", None))
        item = self.tableArgs.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1", None))
        item = self.tableArgs.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "kx", None))
        item = self.tableArgs.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "ky", None))
        item = self.tableArgs.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "kz", None))
        item = self.tableArgs.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "ν", None))
        item = self.tableArgs.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "orts", None))
        item = self.tableArgs.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "D", None))
        item = self.tableArgs.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Ex1", None))
        item = self.tableArgs.horizontalHeaderItem(7)
        item.setText(_translate("MainWindow", "Ey1", None))
        item = self.tableArgs.horizontalHeaderItem(8)
        item.setText(_translate("MainWindow", "Ez1", None))
        item = self.tableArgs.horizontalHeaderItem(9)
        item.setText(_translate("MainWindow", "Nx1", None))
        item = self.tableArgs.horizontalHeaderItem(10)
        item.setText(_translate("MainWindow", "Ny1", None))
        item = self.tableArgs.horizontalHeaderItem(11)
        item.setText(_translate("MainWindow", "Nz1", None))
        item = self.tableArgs.horizontalHeaderItem(12)
        item.setText(_translate("MainWindow", "Ex2", None))
        item = self.tableArgs.horizontalHeaderItem(13)
        item.setText(_translate("MainWindow", "Ey2", None))
        item = self.tableArgs.horizontalHeaderItem(14)
        item.setText(_translate("MainWindow", "Ez2", None))
        item = self.tableArgs.horizontalHeaderItem(15)
        item.setText(_translate("MainWindow", "Nx2", None))
        item = self.tableArgs.horizontalHeaderItem(16)
        item.setText(_translate("MainWindow", "Ny2", None))
        item = self.tableArgs.horizontalHeaderItem(17)
        item.setText(_translate("MainWindow", "Nz2", None))
        __sortingEnabled = self.tableArgs.isSortingEnabled()
        self.tableArgs.setSortingEnabled(False)
        item = self.tableArgs.item(0, 0)
        item.setText(_translate("MainWindow", "sdcs", None))
        item = self.tableArgs.item(0, 1)
        item.setText(_translate("MainWindow", "1", None))
        item = self.tableArgs.item(0, 2)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 3)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 4)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 5)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 6)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 7)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 8)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 9)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 10)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 11)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 12)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 13)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 14)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 15)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 16)
        item.setText(_translate("MainWindow", "3", None))
        item = self.tableArgs.item(0, 17)
        item.setText(_translate("MainWindow", "3", None))
        self.tableArgs.setSortingEnabled(__sortingEnabled)
        self.plus.setText(_translate("MainWindow", "+", None))
        self.loadTable.setText(_translate("MainWindow", "load", None))
        self.tableName.setText(_translate("MainWindow", "default", None))
        self.saveTable.setText(_translate("MainWindow", "Save", None))

