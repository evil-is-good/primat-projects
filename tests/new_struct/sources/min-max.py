import numpy as np

X, Y, T = np.loadtxt('./temperature.gpd', delimiter=' ', usecols=(0, 1, 2), unpack=True)

print T.max(), T.min(), (T.max()-T.min())
