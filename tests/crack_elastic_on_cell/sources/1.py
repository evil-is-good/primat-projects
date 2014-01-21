q  = open('./mata-quadrate.gpd', 'r')
ci = open('./mata-circ.gpd', 'r')
kr = open('./mata-cross.gpd', 'r')
sh = open('./mata-shell.gpd', 'r')
comp = open('./comp.gpd', 'w')

lin = 0
#for line in q:
#	lin = lin + 1

for i in range(28):
	comp.write(str((i+1)**2*16) + ' ' + q.readline().split()[11] + ' '
	 + ci.readline().split()[11] + ' ' + kr.readline().split()[11] + ' '
	 + sh.readline().split()[11] + '\n')

q.close()
ci.close()
kr.close()
sh.close()
comp.close()
