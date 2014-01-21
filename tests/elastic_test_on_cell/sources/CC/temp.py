import os

fi = open('mata-circ.gpd', 'r')

l = fi.read().split("\n")

fi.close()

l.sort()

for i in os.listdir('./'):
	if len(i.split('-')) > 2:
		os.remove(i)

#fo.write(l)

#for i in range(3):
#print l[2].split()[14]
#print l[0]

for i in range(1,len(l)):
	fo = open('mata-circ-' + str(int(float(l[i].split(' ')[0])*1000)) + '.gpd', 'a')
	fo.write(l[i]+'\n')
	fo.close()
	#print l2[i].split()[0]
	#fo.write(l2[i].split()[0] + ' ' + str((float(l2[i].split()[10]) - float(l1[i].split()[10])) / float(l2[i].split()[10])) + '\n')
