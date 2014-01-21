import numpy as np

X, Y, N = np.loadtxt('./res_x.gpd', delimiter=' ', usecols=(0, 1, 2), unpack=True)

fo = open('num1.gpd', 'w')

for x, y, n in zip(X, Y, N)[:len(X) - 1]:
   fo.write('set label ' + '"' + str(int(n)) + '"' + ' at first ' + str(x) + ', first ' + str(y) + '\n')

fo.close()
