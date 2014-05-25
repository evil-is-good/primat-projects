import matplotlib.pyplot as plt
import numpy as np
import math


def gray(x, Min, Max):
    return 1 #(x - Min) / (abs(Min - Max))


def palet1(gray):
    return (1.0 if gray > 0.5 else (gray) * 2.0, \
            (1.0 - gray) * 2.0 if gray > 0.5 else (gray) * 2.0, \
            1.0 if gray < 0.5 else 2.0 * (1 - gray))


def palet2(gray):
    return ((((gray - 0.5) * 8.0) ** 0.5) / 4.0 if gray > 0.5 else 0.0, \
            (gray - 0.5) * 2.0 if gray > 0.5 else 0.0, \
            1.0 if gray > 0.5 else gray * 2.0)

X, Y, S1, S2 = np.loadtxt('main_stress.gpd', delimiter=' ', usecols=(0, 1, 2, 3), unpack=True)
angle1, angle2 = np.loadtxt('angle.gpd', delimiter=' ', usecols=(2, 3), unpack=True)
#MIN, MAX = np.loadtxt('min_max', delimiter=' ', usecols=(0, 1), unpack=True)
#cos2x, cos2y = np.loadtxt('cos_2.gpd', delimiter=' ', usecols=(2, 3), unpack=True)
#X, Y, S1, S2 = np.genfromtxt('main_stress.gpd', delimiter=' ', usecols=(0, 1, 2, 3), unpack=True)

#print MAX[0]

L = 0.016
x1 = 0.0 - L
x2 = 0.0 + L

print S1
print S1.max(), S1.min()

MIN = S1.min()
MAX = S1.max()

fig = plt.figure(num=None, figsize=(10, 5), facecolor='w', edgecolor='k')

for i, j, k, l in zip(X, Y, angle1, S1)[:len(X) - 1]:
    cos = math.cos(k)# - math.pi / 2.0)
    sin = math.sin(k)# - math.pi / 2.0)

    plt.plot([x1 * cos + i, x2 * cos + i], [x1 * sin + j, x2 * sin + j], ls='-', color=palet1(gray(l, MIN, MAX)))
    #plt.plot([i, i], [j, j], 'o', color=(gray(l, MIN, MAX), 0, 0))
    #plt.plot([- 0.2 * l + 0.5, 0.2 * l + 0.5], [- 0.2 * k + 0.5, 0.2 * k + 0.5], 'r-')
    #plt.plot([- 0.01 * k + i, 0.01 * k + i], [- 0.01 * l + j, 0.01 * l + j], 'r-')
    #print i, j, k, l

plt.xlabel('X')
plt.ylabel('Y')
#plt.plot([0.5, 0.5], [0.5, 0.5], linestyle='o')
#plt.plot([-0.5, 0.5], [0.0, 0.0], 'r-', [-0.5, 0.5], [0.3, 0.3], 'b-')
#plt.plot([-0.3, 0.3], [0.2, 0.2], color=(0, 1, 1))
#plt.plot([0.1, 0.1], [0.2, 0.2], 'o')
#plt.show()
ax = plt.axis([0.0, 2.0, 0.0, 1.0])
plt.title('Main places 1 for R = 0.3', fontsize=20, fontname='Serif')
#plt.imshow(S1)
fig.colorbar()
#plt.pcolor(X, Y, S1, vmin=MIN, vmax=MAX)
plt.savefig('main_place_1.png', dpi=100)

plt.clf()

MIN = S2.min()
MAX = S2.max()

for i, j, k, l in zip(X, Y, angle2, S2)[:len(X) - 1]:
    cos = math.cos(k)# - math.pi / 2.0)
    sin = math.sin(k)# - math.pi / 2.0)

    plt.plot([x1 * cos + i, x2 * cos + i], [x1 * sin + j, x2 * sin + j], ls='-', color=palet2(gray(l, MIN, MAX)))

plt.xlabel('X')
plt.ylabel('Y')
#plt.axis([0.0, 1.0, 0.0, 1.0])
plt.title('Main places 2 for R = 0.3', fontsize=20, fontname='Serif')
plt.savefig('main_place_2.png')

#montage main_stress_1.png main_place_1.png -tile 2x1 -geometry +0+0 ms1.png
