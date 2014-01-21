t = '100'

d = '../' + t + '/'

fquadrate = open(d + 'mata-quadrate.gpd', 'r')
fshell = open(d + 'mata-shell.gpd', 'r')
fcross = open(d + 'mata-cross.gpd', 'r')
fcirc = open(d + 'mata-circ.gpd', 'r')
fch = open('../../ch.gpd', 'r')

lq = fquadrate.read().split("\n")
ls = fshell.read().split("\n")
lcr = fcross.read().split("\n")
lci = fcirc.read().split("\n")
lch = fch.read().split("\n")

fquadrate.close()
fshell.close()
fcross.close()
fcirc.close()
fch.close()

num = [[lq, 'q'], [ls, 's'], [lcr, 'cr'], [lci, 'ci']]
cols = [1, 2, 3, 7, 9, 10, 11]

fo = open(t + '_max.gpd', 'w')

for m in num:
    maxim = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i, j in zip(m[0], lch)[:len(m[0]) - 1]:
        k = 0
        for n in cols:
            a = abs(float(i.split()[n]) - float(j.split()[n])) / float(i.split()[n])
            if (a > maxim[k]):
                maxim[k] = a
            k += 1
    fo.write(m[1] + ' ' + str(maxim) + '\n')

fo.close()
