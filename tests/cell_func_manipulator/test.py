f = open("test.bin", "rb")
r = f.read()
f.close()
# for x in r:
# 	print x

import struct
import binascii

# values = (1, 'ab', 2.7)
# s = struct.Struct('I 2s f')
# packed_data = s.pack(*values)
#
# print 'Original values:', values
# print 'Format string  :', s.format
# print 'Uses           :', s.size, 'bytes'
# print 'Packed Value   :', binascii.hexlify(packed_data)
# s = struct.Struct('I I I I I I')
s = struct.Struct(' '.join(['d' for i in xrange(6)]))
u_arr = s.unpack(r)
print type(r), type(u_arr), u_arr

import numpy as np

a = np.array(u_arr, np.double)

print a 

import matplotlib.pyplot as plt

# plt.plot(a)
# plt.show()

f = open("/home/primat/projects/tests/new_struct/sources/cell/E_10_R_25_4/solution_on_cell_size.bin")
size = f.read()
f.close()
size = struct.Struct('I I').unpack(size)[0]
f = open("/home/primat/projects/tests/new_struct/sources/cell/E_10_R_25_4/stress_1_0_0_x_x.bin")
stress = f.read()
f.close()
stress = struct.Struct(' '.join(['d' for i in xrange(size)])).unpack(stress)
print size, len(stress)
stress = np.array(
        [[stress[i] for i in xrange(0,size,3)],
        [stress[i] for i in xrange(1,size,3)],
        [stress[i] for i in xrange(2,size,3)]],
        np.double)
print len(stress[0])
f = open("/home/primat/projects/tests/new_struct/sources/cell/E_10_R_25_4/coor_cell.bin")
coor = f.read()
f.close()
coor = struct.Struct(' '.join(['d' for i in xrange(size*3)])).unpack(coor)
print size, len(coor)
print [coor[i] for i in xrange(18)]
coor = np.array(
        [[[coor[i] for i in xrange(0,size*3,3)][j] for j in xrange(0,size,3)],
        [[coor[i] for i in xrange(1,size*3,3)][j] for j in xrange(0,size,3)],
        [[coor[i] for i in xrange(2,size*3,3)][j] for j in xrange(0,size,3)]],
        np.double)
# coor = np.array(
#         [[coor[i] for i in xrange(0,9,3)],
#         [coor[i] for i in xrange(1,9,3)],
#         [coor[i] for i in xrange(2,9,3)]],
#         np.double)
# coor = [[i for i in xrange(0,9,3)], [i for i in xrange(1,9,3)], [i for i in xrange(2,9,3)]]

print len(coor[0]),len(coor[1]),len(coor[2])

size = size / 3

line = [stress[0][i] for i in xrange(size) if ((abs(coor[1][i]-0.5) < 1e-10) and (abs(coor[2][i]-0.5) < 1e-10))]

plt.plot(line)
plt.show()
