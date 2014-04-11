dir = '../10/'

fquadrate = open(dir + 'mata-quadrate.gpd', 'r')
fshell = open(dir + 'mata-shell.gpd', 'r')
fcross = open(dir + 'mata-cross.gpd', 'r')
fcirc = open(dir + 'mata-circ.gpd', 'r')
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

fo = open('comparison_q_ch.gpd', 'w')
for i, j in zip(lq, lch)[:len(lq) - 1]:
    s = str(float(i.split()[0]) / 16384) + ' '
    for k, l in zip(i.split(), j.split())[1:]:
        s += str(abs((float(k) - float(l))) / float(k)) + ' '
    fo.write(s + '\n')

fo.close()

#print zip(ls, lch)[:len(ls) - 2]

fo = open('comparison_s_ch.gpd', 'w')
for i, j in zip(ls, lch)[:len(ls) - 1]:
    s = str(float(i.split()[0]) / 16384) + ' '
    for k, l in zip(i.split(), j.split())[1:]:
        s += str(abs((float(k) - float(l))) / float(k)) + ' '
    fo.write(s + '\n')

fo.close()

fo = open('comparison_cr_ch.gpd', 'w')
for i, j in zip(lcr, lch)[:len(lcr) - 1]:
    s = str(float(i.split()[0]) / 16384) + ' '
    for k, l in zip(i.split(), j.split())[1:]:
        s += str(abs((float(k) - float(l))) / float(k)) + ' '
    fo.write(s + '\n')

fo.close()

fo = open('comparison_ci_ch.gpd', 'w')
for i, j in zip(lci, lch)[:len(lci) - 1]:
    s = str(float(i.split()[0]) / 16384) + ' '
    for k, l in zip(i.split(), j.split())[1:]:
        s += str(abs((float(k) - float(l))) / float(k)) + ' '
    fo.write(s + '\n')

fo.close()
