import numpy as np

X, Y, N = np.loadtxt('./grad_x_xx.gpd', delimiter=' ', usecols=(0, 1, 3), unpack=True)

fo = open('num.gpd', 'w')

for x, y, n in zip(X, Y, N)[:len(X) - 1]:
   fo.write('set label ' + '"' + str(int(n)) + '"' + ' at first ' + str(x) + ', first ' + str(y) + '\n')

fo.close()
