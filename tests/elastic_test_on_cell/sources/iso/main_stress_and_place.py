import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
from matplotlib.colors import LinearSegmentedColormap


# type_cell = "hex"
type_cell = "tetr"

for nn in xrange(41):
    plt.clf()
    xs, ys, ms1, ms2, tang = np.loadtxt(type_cell+'/6/main_stress_and_place'+str(nn)+'.nogit', delimiter=' ', usecols=(0, 1, 2, 3, 6), unpack=True)
    xa, ya, a1, a2 = np.loadtxt(type_cell+'/5/main_stress_and_place'+str(nn)+'.nogit', delimiter=' ', usecols=(0, 1, 4, 5), unpack=True)
# 
    Zmax, Zmin = ms1.max(), ms1.min()

    if Zmax < 0.0:
        cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

              'blue':  ((0.0, 0.0, 1.0),
                       (1.0, 1.0, 1.0),
                       (1.0, 1.0, 0.0))
             }
    else:
        if Zmin > 0.0:
            cdict1 = {'red':   ((0.0, 0.0, 1.0),
                       (0.0, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

              'green': ((0.0, 0.0, 1.0),
                       (0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),

              'blue':  ((0.0, 0.0, 1.0),
                       (0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0))
             }
        else:
            Zmid = -Zmin / (Zmax - Zmin) 
            cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (Zmid, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                       (Zmid, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),

              'blue':  ((0.0, 0.0, 1.0),
                       (Zmid, 1.0, 1.0),
                       (1.0, 0.0, 0.0))
             }


    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

    xg = np.linspace(xs.min(),xs.max(),int(len(xs)**0.5))
    yg = np.linspace(ys.min(),ys.max(),int(len(xs)**0.5))
    X,Y = np.meshgrid(xg,yg)
# 
# interpolate Z values on defined grid
    Z = interpolate.griddata(np.vstack((xs.flatten(),ys.flatten())).T, \
    np.vstack(ms1.flatten()),(X,Y),method='cubic').reshape(X.shape)
    Tang = interpolate.griddata(np.vstack((xs.flatten(),ys.flatten())).T, \
    np.vstack(tang.flatten()),(X,Y),method='cubic').reshape(X.shape)
# mask nan values, so they will not appear on plot
# Zm = np.ma.masked_where(np.isnan(Z),Z)
# 
    plt.pcolormesh(X, Y, Z, cmap=blue_red1)
    plt.colorbar()
# 
# 
    print nn
    L = 0.016
    x1 = 0.0 - L
    x2 = 0.0 + L
    for i, j, k in zip(xa, ya, a1)[:len(xa) - 1]:
        cos = math.cos(k - math.pi / 2.0)
        sin = math.sin(k - math.pi / 2.0)

        plt.plot([x1 * cos + i, x2 * cos + i], [x1 * sin + j, x2 * sin + j], ls='-', color=(0.0, 0.0, 0.0))
    
    plt.xlabel('X')
    plt.ylabel('Y')
    # ax = plt.axis([0.0, 2.0, 0.0, 3.0**0.5])
    ax = plt.axis([0.0, 1.0, 0.0, 1.0])
    
    if nn > 9:
        name = str(nn)
    else:
        name = '0' + str(nn)
    plt.savefig(type_cell+'/clip2/main_stress_and_place_1_0'+name+'.png', dpi=100)

    plt.clf()

    plt.pcolormesh(X, Y, Tang, cmap=blue_red1)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    # ax = plt.axis([0.0, 2.0, 0.0, 3.0**0.5])
    ax = plt.axis([0.0, 1.0, 0.0, 1.0])
    for i, j, k in zip(xa, ya, a1)[:len(xa) - 1]:
        cos = math.cos(k - math.pi / 4.0)
        sin = math.sin(k - math.pi / 4.0)

        plt.plot([x1 * cos + i, x2 * cos + i], [x1 * sin + j, x2 * sin + j], ls='-', color=(0.0, 0.0, 0.0))
    plt.savefig(type_cell+'/clip2/main_tangens_0'+name+'.png', dpi=100)


# xs, ys, ms1 = np.loadtxt('test_for_py.nogit', delimiter=' ', usecols=(0, 1, 2), unpack=True)
# 
# print xs
# 
# xg = np.linspace(xs.min(),xs.max(),2)
# yg = np.linspace(ys.min(),ys.max(),2)
# 
# print xg
# 
# X,Y = np.meshgrid(xg,yg)
# 
# print X
