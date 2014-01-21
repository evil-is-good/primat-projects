#import matplotlib.pyplot as plt
import numpy as np
#import math

fo = open("min-max", 'w')

for i in range(0, 51, 1):
    X, Y, S1, S2 = np.loadtxt('main_stress' + str(i) + '.gpd', delimiter=' ', usecols=(0, 1, 2, 3), unpack=True)

    fo.write(str(i / 100.0) + ' ' + str(Y[np.argmax(S1)]) + ' ' +
            str(X[np.argmax(S1)]) + ' ' + str(S1.max()) + ' ' + str(S2.min()) + ' ')

    max_y = 0.0
    max_s = 0.0
    for x, y, s in zip(X, Y, S1)[:len(X) - 1]:
        #print x, abs(x - 0.5)
        if x == 0.5:
            #abs(x - 0.5) < 0e-12:
            #print x, y, s
            if s > max_s:
                max_y = y
                max_s = s
    fo.write(str(max_y) + ' ' + str(max_s) + ' ')

    max_y = 0.0
    max_s = 0.0
    for x, y, s in zip(X, Y, S1)[:len(X) - 1]:
        if x == y:
            #abs(x - y) < 0e-12:
            if s > max_s:
                max_y = y
                max_s = s
    fo.write(str(max_y) + ' ' + str(max_s) + ' ')

    for x, y, s in zip(X, Y, S1)[:len(X) - 1]:
        if (x == 0.5) and (y == 0.5):
            fo.write(str(s) + '\n')
            break
    #MAX.append(S.max())

fo.close()

#S1_x, S1_y, S1_max, x_05, max_05, y_x, max_y_x, max_c = np.loadtxt("min-max",  delimiter=' ', usecols=(1, 2, 3, 5, 6, 7, 8, 9), unpack=True)
#
#print S1_max
#
#fo = open("main_stress_1_all.g", 'w')
#fo.write('reset\nset xrange [0:1.0]\nset yrange [0:1.0]\nset xtics 0, 0.1, 1.0\n')
#fo.write('set ytics 0,0.1,1.0\nset size 1.0, 1.0\nset term png enhanced size 1024, 1024\n')
#fo.write('unset key\nset pm3d map\nset xlabel "X"\nset ylabel "Y"\n')
#fo.write('set palette function gray > 0.5 ? 1.0 : (gray) * 2.0, gray > 0.5 ? (1.0 - gray) * 2.0: (gray) * 2.0, gray < 0.5 ? 1.0 : 2.0 * (1 - gray)')
#
#fo.write('\n\n')
#
#for i in range(10):
#    fo.write('set title "Stress {/Symbol s}_1 for R = ' + str((i + 1) * 0.05) + '" font "Serif, 30"\n')
#    fo.write('set label "A' + '" at first ' + str(S1_x[i]) +
#            ', first ' + str(S1_y[i]) + ' font "Symbol,20" front point pt 7 ps 2 offset 1 \n')
#    fo.write('set label "A = ' + str(round(S1_max[i], 3)) + '" at first ' + str(0.1) +
#            ', first ' + str(0.95) + ' font "Symbol,20" front point pt 7 ps 2 offset 1 \n')
#    #fo.write('set label "B:' + str(round(max_y_x[i], 3)) + '" at first ' + str(y_x[i]) +
#    #        ', first ' + str(y_x[i]) + ' font "Symbol,20" front point pt 7 ps 2 offset 1 \n')
#    fo.write('set label "B' + '" at first ' + str(x_05[i]) +
#            ', first ' + str(0.5) + ' font "Symbol,20" front point pt 7 ps 2 offset 1 \n')
#    fo.write('set label "B = ' + str(round(max_05[i], 3)) + '" at first ' + str(0.1) +
#            ', first ' + str(0.90) + ' font "Symbol,20" front point pt 7 ps 2 offset 1 \n')
#    #fo.write('set label "C:' + str(max_c[i]) + '" at first ' + str(0.5) +
#    #        ', first ' + str(0.5) + ' font "Symbol,20" front \n')
#    fo.write('set output "main_stress' + str((i + 1) * 5) + '_1.png"\n')
#    fo.write('splot "main_stress' + str((i + 1) * 5) + '.gpd" using ($2):($1):($3) with pm3d\n')
#    fo.write('set nolabel\n\n')
#
#fo.close()
#
#a = np.array([1, 7, 10, 4, 5])
#print a, np.argmax(a)








#print 'AAAV'

#X, Smin, Smax = np.loadtxt('min-max', delimiter=' ', usecols=(0, 1, 2), unpack=True)

#for i in range(0, X.size(), 1):
#    X[i] = X[i] / 10.0

#plt.xlabel('X');
#plt.ylabel('Y');

#plt.plot(X, Smax, "r-")
#plt.axis(5.0, 55.0, [0.6, 0.8, 1.0]

#plt.savefig('max_stress_1.png')
