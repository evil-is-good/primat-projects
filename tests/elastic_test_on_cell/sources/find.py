import numpy as np

heat = np.loadtxt('../../heat_test/sources/matrix.gpd', delimiter=' ')
cell = np.loadtxt('../../cell_heat_test/sources/matrix_cell.gpd', delimiter=' ')

fo = open('check.gpd', 'w')

for i in heat:
    for j in cell:
        if (int(i[0]) == int(j[0])) and (int(i[1]) == int(j[1])):
            fo.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(abs(i[2] - j[2])) + '\n')
