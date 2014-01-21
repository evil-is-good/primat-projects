import numpy as np

#A = np.loadtxt('../../heat_test/sources/matrix.gpd', delimiter=' ', usecols=(2,))
#A = np.reshape(A, (int(A.size ** 0.5), -1))

#A = np.loadtxt('../../heat_test/sources/A.gpd', delimiter=' ')
#b = np.loadtxt('../../heat_test/sources/b.gpd', delimiter=' ')

A = np.loadtxt('../../cell_heat_test/sources/A.gpd', delimiter=' ')
b = np.loadtxt('../../cell_heat_test/sources/b.gpd', delimiter=' ')

print A.size ** 0.5
print b.size

#np.savetxt('A', A)

x = np.linalg.solve(A, b)

print 'A'
print A
print 'b'
print b
print 'x'
print x

xdeal = np.loadtxt('../../cell_heat_test/sources/x.gpd', delimiter=' ')

print 'xdeal'
print xdeal

print 'x_sub'
print np.subtract(x, xdeal)

#cell = np.loadtxt('../../cell_heat_test/sources/matrix_cell.gpd', delimiter=' ')

#fo = open('check.gpd', 'w')

#for i in heat:
#    for j in cell:
#        if (int(i[0]) == int(j[0])) and (int(i[1]) == int(j[1])):
#            fo.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(abs(i[2] - j[2])) + '\n')
