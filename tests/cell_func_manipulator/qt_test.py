# -*- coding: utf-8 -*-
import struct
import binascii
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from PySide import QtGui
from PySide import QtCore
from PySide.QtGui import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle(u'Главное окно')    # если устанавливать значение не unicode, то будут печататься крокозябры
        self.resize(300, 200)
        self.cw = QtGui.QWidget()  # на главном окне нужно определить central widget
        self.layout = QtGui.QGridLayout()  # у central widget должна быть определена разметка, чтобы добавлять в неё gui-элементы
        self.cw.setLayout(self.layout)
        self.setCentralWidget(self.cw)

        self.textField = QtGui.QLineEdit(self)
        self.textField.setText('xx')
        self.layout.addWidget(self.textField, 0, 0)

        self.button1 = QtGui.QPushButton()
        self.button1.setText(u'Отрисовка графика')
        self.button1.clicked.connect(self.showGraph)
        self.layout.addWidget(self.button1, 1, 0)

        # self.keyPressEvent.connect(self.showGraph1)
        
        # self.key = QtGui.QKeyEvent(QtGui.QEvent.KeyPress, 0

        self.coefs = [QtGui.QLineEdit(self) for i in xrange(16)]
        self.coefs[0].setText('2')
        self.coefs[1].setText('1.0')
        self.coefs[2].setText('1.0')
        self.coefs[3].setText('1.0')
        self.coefs[4].setText('0.25')
        self.coefs[5].setText('0.25')
        self.coefs[6].setText('0.25')
        self.coefs[7].setText('10.0')
        self.coefs[8].setText('10.0')
        self.coefs[9].setText('10.0')
        self.coefs[10].setText('0.25')
        self.coefs[11].setText('0.25')
        self.coefs[12].setText('0.25')
        self.coefs[13].setText('2')
        self.coefs[14].setText('3')
        self.coefs[15].setText('0.25')
        for i in xrange(16):
            self.layout.addWidget(self.coefs[i], i+2, 0)

        self.button2 = QtGui.QPushButton()
        self.button2.setText(u'Расчёт')
        self.button2.clicked.connect(self.calc)
        self.layout.addWidget(self.button2, 18, 0)

        self.memo = QtGui.QPlainTextEdit(self)
        self.memo.setPlainText('xxsdsdsd')
        self.layout.addWidget(self.memo, 19, 0)
 
    def showGraph(self):
        path = "/home/primat/hoz_block_disk/cell_func_bd/2/1.0_1.0_1.0_0.25_0.25_0.25/10.0_10.0_10.0_0.25_0.25_0.25/fiber/circle_pixel/0.25/3/"
        f = open(path + "size_solution.bin")
        size = struct.Struct('I I').unpack(f.read())[0]
        f.close()

        pattern = ' '.join(['d' for i in xrange(size)])

        x, y, z = 0, 1, 2
        ort = ['x', 'y', 'z']

        coor = [0,0,0]
        for i in x,y,z:
            f = open(path + "coor_" + ort[i] + ".bin")
            coor[i] = struct.Struct(pattern).unpack(f.read())
            f.close()

        f = open(path + "1/1_0_0/x/stress/"+self.textField.displayText()+".bin")
        stress = struct.Struct(pattern).unpack(f.read())
        f.close()

        line = [stress[i] for i in xrange(size) if ((abs(coor[y][i]-0.5) < 1e-10) and (abs(coor[z][i]-0.5) < 1e-10))]
        plt.plot(line)
        plt.show()
 
    # def showGraph1(self, e):
    #     path = "/home/primat/hoz_block_disk/cell_func_bd/2/1.0_1.0_1.0_0.25_0.25_0.25/10.0_10.0_10.0_0.25_0.25_0.25/fiber/circle_pixel/0.25/3/"
    #     f = open(path + "size_solution.bin")
    #     size = struct.Struct('I I').unpack(f.read())[0]
    #     f.close()
    #
    #     pattern = ' '.join(['d' for i in xrange(size)])
    #
    #     x, y, z = 0, 1, 2
    #     ort = ['x', 'y', 'z']
    #
    #     coor = [0,0,0]
    #     for i in x,y,z:
    #         f = open(path + "coor_" + ort[i] + ".bin")
    #         coor[i] = struct.Struct(pattern).unpack(f.read())
    #         f.close()
    #
    #     f = open(path + "1/1_0_0/x/stress/"+self.textField.displayText()+".bin")
    #     stress = struct.Struct(pattern).unpack(f.read())
    #     f.close()
    #
    #     line = [stress[i] for i in xrange(size) if ((abs(coor[y][i]-0.5) < 1e-10) and (abs(coor[z][i]-0.5) < 1e-10))]
    #     plt.plot(line)
    #     plt.show()

    def calc(self):
        arguments = ' '.join([i.displayText() for i in self.coefs])
        proc = subprocess.Popen("cd ~/projects/work/elastic_cell_calculator/sources/; make clean; make; ../release/elastic_cell_calculator.exe "
                + arguments, shell=True, stdout=subprocess.PIPE)
        self.memo.setPlainText(' '.join(proc.stdout.readlines()))
        # self.memo.setPlainText(arguments)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    sys.exit( app.exec_() )
