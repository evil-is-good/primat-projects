# -*- coding: utf-8 -*-
import struct
import binascii
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import QApplication, QMainWindow
import cfm_ui

def print_arg(*args):
    print type(args)
    print args

class CFM(QMainWindow, cfm_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        f = open("default.tbl")
        a = f.read().split()
        f.close()
        for i in xrange(1,19):
            self.tableArgs.item(0,i-1).setText(a[i])
        self.plot_stress.clicked.connect(self.showPlotStress)
        self.plot_deform.clicked.connect(self.showPlotDeform)
        self.plot_move.clicked.connect(self.showPlotMove)
        self.plus.clicked.connect(self.insertRow)
        self.loadTable.clicked.connect(self.loadTable_s)
        self.saveTable.clicked.connect(self.saveTable_s)

    def showPlot(self, what):
        # path = ("/home/primat/hoz_block_disk/cell_func_bd"
        #         + "/" + self.num_mat.displayText()
        #         + "/"
        #         +       self.Ex1.displayText()
        #         + "_" + self.Ey1.displayText()
        #         + "_" + self.Ez1.displayText()
        #         + "_" + self.Nx1.displayText()
        #         + "_" + self.Ny1.displayText()
        #         + "_" + self.Nz1.displayText()
        #         + "/"
        #         +       self.Ex2.displayText()
        #         + "_" + self.Ey2.displayText()
        #         + "_" + self.Ez2.displayText()
        #         + "_" + self.Nx2.displayText()
        #         + "_" + self.Ny2.displayText()
        #         + "_" + self.Nz2.displayText()
        #         + "/" + "fiber/circle_pixel"
        #         + "/" + self.D.displayText()
        #         + "/" + self.num_ref.displayText())
        num_plots = self.tableArgs.rowCount()
        line_coor = [[] for i in range(num_plots)]
        line = [[] for i in range(num_plots)]
        for n in range(num_plots):
            # path = ("/home/primat/hoz_block_disk/cell_func_bd"
            path = ("/media/primat/b9ef6a43-441a-458d-9f0a-76b1042d57fb/cell_func_bd"
                    + "/" + self.num_mat.displayText()
                    + "/" + "_".join([str(self.tableArgs.item(n,i).text()) for i in range(6,12)])
                    + "/" + "_".join([str(self.tableArgs.item(n,i).text()) for i in range(12,18)])
                    + "/" + "fiber/circle_pixel"
                    + "/" + self.tableArgs.item(n,5).text()
                    + "/" + self.num_ref.displayText()
                    )

            f = open(path + "/size_solution.bin")
            size = struct.Struct('I I').unpack(f.read())[0]
            f.close()

            pattern = ' '.join(['d' for i in xrange(size)])

            x, y, z = 0, 1, 2
            ort = ['x', 'y', 'z']

            coor = [0,0,0]
            for i in x,y,z:
                f = open(path + "/coor_" + ort[i] + ".bin")
                coor[i] = struct.Struct(pattern).unpack(f.read())
                f.close()

            file_name = (path
                    + "/" + "1"
                    + "/" + "_".join([str(self.tableArgs.item(n,i).text()) for i in range(0,3)])
                    + "/" + self.tableArgs.item(n,3).text()
                    + "/" + what
                    + "/" + self.tableArgs.item(n,4).text())

            f = open(file_name+".bin")
            solution = struct.Struct(pattern).unpack(f.read())
            f.close()

            line[n] = [solution[i] for i in xrange(size) if ((abs(coor[y][i]-0.5) < 1e-10) and (abs(coor[z][i]-0.5) < 1e-10))]
            line_coor[n] = [coor[x][i] for i in xrange(size) if ((abs(coor[y][i]-0.5) < 1e-10) and (abs(coor[z][i]-0.5) < 1e-10))]

        plt.plot(*[(line_coor[i/2] if ((i % 2) == 0) else line[i/2]) for i in xrange(num_plots*2)])
        plt.show()
 
    def showPlotStress(self):
        self.showPlot("stress")
 
    def showPlotDeform(self):
        self.showPlot("deform")
 
    def showPlotMove(self):
        self.showPlot("move")

    def calc(self):
        arguments = ' '.join([i.displayText() for i in self.coefs])
        proc = subprocess.Popen(
                "cd ~/projects/work/elastic_cell_calculator/sources/; make clean;"
                + "make; ../release/elastic_cell_calculator.exe "
                + arguments, shell=True, stdout=subprocess.PIPE)
        self.memo.setPlainText(' '.join(proc.stdout.readlines()))

    def insertRow(self):
        self.tableArgs.insertRow(self.tableArgs.rowCount())
        num_rows = self.tableArgs.rowCount()-1
        # print num_rows
        # print self.tableArgs.currentColumn()
        f = open("default.tbl")
        a = f.read().split()
        f.close()
        for i in xrange(1,19):
            self.tableArgs.setItem(num_rows, i-1, QtGui.QTableWidgetItem())
            self.tableArgs.item(num_rows, i-1).setText(a[i])

    def loadTable_s(self):
        f = open(self.tableName.displayText() + ".tbl")
        a = f.read().split()
        f.close()
        b = [[a[j] for j in xrange(18*i+1, 18*i+19)] for i in xrange(int(a[0]))]

        for i in xrange(len(b)):
            for j in xrange(0,18):
                self.tableArgs.item(i,j).setText(b[i][j])

    def saveTable_s(self):
        b = [[self.tableArgs.item(i,j).text() for j in xrange(18)] for i in xrange(self.tableArgs.rowCount())]
        f = open(self.tableName.displayText() + ".tbl", 'w')
        f.write(str(len(b)) + '\n')
        for i in xrange(len(b)):
            for j in xrange(0,18):
                f.write(b[i][j] + '\n')
        f.close()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = CFM()
    form.show()
    app.exec_()
    # print [(i if ((i % 2) == 0) else i+1) for i in xrange(10)]
    # print_arg([10, 20], [30, 20])
    # print_arg(*[ttt(i,i+1) for i in range(10)])
