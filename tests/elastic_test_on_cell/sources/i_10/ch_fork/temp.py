fi1 = open('mata-circ_for_hex.gpd', 'r')
fi2 = open('mata-hexagon_for_cir.gpd', 'r')
fo  = open('hex_and_circ.gpd', 'w')

l1 = fi1.read().split("\n")
l2 = fi2.read().split("\n")

fi1.close()
fi2.close()

for i in range(len(l1)-1):
	#print l2[i].split()[0]
	fo.write(l2[i].split()[0] + ' ' + str((float(l2[i].split()[10]) - float(l1[i].split()[10])) / float(l2[i].split()[10])) + '\n')
