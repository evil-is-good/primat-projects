#from numpy import reshape
import numpy as np
from pylab import *

NX=80
NY=80

# read gnuplot x-y-z datafile, skip empty lines
#d=[i.strip().split() for i in
#	open("sigma_y").readlines()
#	if len(i.strip()) > 0]
# tranpose the list of lists
#d=zip(*d)
# convert strings to floats
#X=[float(i) for i in d[0]]
#Y=[float(i) for i in d[1]]
#Z=[float(i) for i in d[2]]
# reshape them to be 2D arrays

# finally plot it\

X, Y, S = np.loadtxt ('sigma_y.gpd', delimiter = ' ', usecols = (0, 1, 3), unpack = True)

print S

X = np.reshape (X, (int(X.size**0.5), -1))
Y = np.reshape (Y, (int(X.size**0.5), -1))
S = np.reshape (S, (int(X.size**0.5), -1))

print S.size

cset = contour(X, Y, S, colors = 'black')

clabel (cset, fmt="%1.1f", fontsize = 9)

savefig ('pylab.png')
