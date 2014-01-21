fj = open("jancovsky.gpd", 'r')
fh = open("meta-hex.gpd", 'r')
fo = open("div_hex-janc.gpd", 'w')

for i, j in zip(fj, fh):
    ja = [float(k) for k in i.split()]
    he = [float(k) for k in j.split()]
    fo.write("{0} {1} {2} {3} {4}\n".format(
        ja[0],
        abs(he[1] - ja[1]) / he[1],
        abs(he[1] - ja[3]) / he[1],
        abs(he[1] - ja[2]) / he[1],
        abs(he[1] - ja[4]) / he[1]))

fj.close()
fh.close()
fo.close()

fj = open("jancovsky_cuprum.gpd", 'r')
fh = open("meta-hex_cuprum.gpd", 'r')
fo = open("div_hex-janc_cuprum.gpd", 'w')

for i, j in zip(fj, fh):
    ja = [float(k) for k in i.split()]
    he = [float(k) for k in j.split()]
    fo.write("{0} {1} {2} {3} {4}\n".format(
        ja[0],
        abs(he[1] - ja[1]) / he[1],
        abs(he[1] - ja[3]) / he[1],
        abs(he[1] - ja[2]) / he[1],
        abs(he[1] - ja[4]) / he[1]))

fj.close()
fh.close()
fo.close()

fj = open("jancovsky_cuprum.gpd", 'r')
fh = open("meta-hex_cuprum.gpd", 'r')
fo = open("div_hex-janc_cuprum_midle.gpd", 'w')

for i, j in zip(fj, fh):
    ja = [float(k) for k in i.split()]
    he = [float(k) for k in j.split()]
    fo.write("{0} {1} {2} {3}\n".format(
        ja[0],
        abs(he[1] - ((ja[1] + ja[3]) / 2.0)) / he[1],
        abs(he[1] - ((ja[2] + ja[4]) / 2.0)) / he[1],
        abs(he[1] - ((ja[1] + ja[4]) / 2.0)) / he[1]))

fj.close()
fh.close()
fo.close()
