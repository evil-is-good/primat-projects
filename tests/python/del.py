import os

#for i in os.listdir('./'):
#	if len(i.split('-')) > 2:
#		os.remove(i)

def ls_r (path):
	res = []
	ls = os.listdir(path)
	#print path, ls
	for i in ls:
		if os.path.isdir(os.path.join(path,i)):
			res += ls_r (os.path.join(path, i))
			#print i
		else:
			#print 'no_dir: ', i
			res.append(os.path.join(path, i))
	return res
					
#print os.listdir('./')

ls = ls_r('./tests')#os.path.join('./', 
#print ls

binfiles = []
for i in ls:
	if i.split('.')[-1] == 'exe' or i.split('.')[-1] == 'o' or i.split('.')[-1] == 'out' or i.split('.')[-1] == 'gch':
		binfiles.append(i)
print binfiles

#for i in binfiles:
#	os.remove(i)

print os.path.isdir('./tests/tex')

#if os.path.isdir(os.listdir('./')[3]):
#        print os.listdir('./')[3]

