import math

g = open('mata-hexagon_g.gpd', 'r')
ch = open('ch_fork.gpd', 'r')

for i in range(12):
	g.readline()

for i in range(10):
	ch.readline()

a = g.readline().split()
b = ch.readline().split()

g.close()
ch.close()

c = [float(a[12]), float(a[10]), float(a[3]), float(a[9])]
d = [float(b[12]), float(b[10]), float(b[3]), float(b[9])]

print ((c[0] - c[1] - 4.0 * c[2]*c[2] * c[0] * c[1] / c[3]) / 
	(c[0] + c[1] + 4.0 * c[2]*c[2] * c[0] * c[1] / c[3]))
	
print float(b[2])

print d[0] - c[0], d[1] - c[1], d[2] - c[2], d[3] - c[3]

print (c[0] - c[1] - 4.0 * c[2]*c[2] * c[0] * c[1] / c[3])

print (c[0] + c[1] + 4.0 * c[2]*c[2] * c[0] * c[1] / c[3])
