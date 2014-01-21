q  = open('./mata-quadrate.gpd', 'r')
ci = open('./mata-circ.gpd', 'r')
kr = open('./mata-cross.gpd', 'r')
sh = open('./mata-shell.gpd', 'r')
comp_c = open('./comp_c.gpd', 'w')
comp_cs = open('./comp_cs.gpd', 'w')

lin = 0
#for line in q:
#	lin = lin + 1

for i in range(28):
	i_g = float(q.readline().split()[11])
	i_ci = float(ci.readline().split()[11])
	i_kr = float(kr.readline().split()[11])
	i_sh = float(sh.readline().split()[11])
	comp_c.write(str((i+1)**2*16.0/16384.0) + ' ' + str(i_g) + ' ' + str(i_ci) + '\n')
	comp_cs.write(str((i+1)**2*16.0/16384.0) + ' ' + str(i_g) + ' ' + str(i_kr)
	+ ' ' + str(i_sh) + '\n')
	#comp.write(str((i+1)**2*16) + ' ' + q.readline().split()[11] + ' '
	# + ci.readline().split()[11] + ' ' + kr.readline().split()[11] + ' '
	# + sh.readline().split()[11] + '\n')
for i in range(3):
	i_g = float(q.readline().split()[11])
	i_kr = float(kr.readline().split()[11])
	i_sh = float(sh.readline().split()[11])
	comp_cs.write(str((i+29)**2*16.0/16384.0) + ' ' + str(i_g) + ' ' + str(i_kr)
	+ ' ' + str(i_sh) + '\n')

q.close()
ci.close()
kr.close()
sh.close()
comp_c.close()
comp_cs.close()
