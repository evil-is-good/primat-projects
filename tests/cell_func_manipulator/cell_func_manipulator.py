import struct
import binascii

import numpy as np

import matplotlib.pyplot as plt

# plt.plot(a)
# plt.show()

path = "/home/primat/hoz_block_disk/cell_func_bd/2/1.0_1.0_1.0_0.25_0.25_0.25/10.0_10.0_10.0_0.25_0.25_0.25/fiber/circle_pixel/0.25/3/"

f = open(path + "size_solution.bin")
size = struct.Struct('I I').unpack(f.read())[0]
f.close()
print size

pattern = ' '.join(['d' for i in xrange(size)])

x, y, z = 0, 1, 2
ort = ['x', 'y', 'z']

coor = [0,0,0]
for i in x,y,z:
    f = open(path + "coor_" + ort[i] + ".bin")
    coor[i] = struct.Struct(pattern).unpack(f.read())
    f.close()

f = open(path + "1/1_0_0/x/stress/xx.bin")
stress = struct.Struct(pattern).unpack(f.read())
f.close()

line = [stress[i] for i in xrange(size) if ((abs(coor[y][i]-0.5) < 1e-10) and (abs(coor[z][i]-0.5) < 1e-10))]
print line


# f = open("/home/primat/projects/tests/new_struct/sources/cell/E_10_R_25_4/stress_1_0_0_x_x.bin")
# stress = f.read()
# f.close()
# stress = struct.Struct(' '.join(['d' for i in xrange(size)])).unpack(stress)
# print size, len(stress)
# stress = np.array(
#         [[stress[i] for i in xrange(0,size,3)],
#         [stress[i] for i in xrange(1,size,3)],
#         [stress[i] for i in xrange(2,size,3)]],
#         np.double)
# print len(stress[0])
# f = open("/home/primat/projects/tests/new_struct/sources/cell/E_10_R_25_4/coor_cell.bin")
# coor = f.read()
# f.close()
# coor = struct.Struct(' '.join(['d' for i in xrange(size*3)])).unpack(coor)
# print size, len(coor)
# print [coor[i] for i in xrange(18)]
# coor = np.array(
#         [[[coor[i] for i in xrange(0,size*3,3)][j] for j in xrange(0,size,3)],
#         [[coor[i] for i in xrange(1,size*3,3)][j] for j in xrange(0,size,3)],
#         [[coor[i] for i in xrange(2,size*3,3)][j] for j in xrange(0,size,3)]],
#         np.double)
# # coor = np.array(
# #         [[coor[i] for i in xrange(0,9,3)],
# #         [coor[i] for i in xrange(1,9,3)],
# #         [coor[i] for i in xrange(2,9,3)]],
# #         np.double)
# # coor = [[i for i in xrange(0,9,3)], [i for i in xrange(1,9,3)], [i for i in xrange(2,9,3)]]
#
# print len(coor[0]),len(coor[1]),len(coor[2])
#
# size = size / 3
#
# line = [stress[0][i] for i in xrange(size) if ((abs(coor[1][i]-0.5) < 1e-10) and (abs(coor[2][i]-0.5) < 1e-10))]
#
plt.plot(line)
plt.show()
