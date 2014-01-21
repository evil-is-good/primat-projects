reset

mean(size,l1,l2) = ((size*l1+(128.0-size)*l2)/128.0)
serial(size,l1,l2) = 128.0*l1*l2/(size*l2+(128.0-size)*l1)
derak(x)  = x == 6400?x:-1000000
forkS(x) = x/128.0*(1.0/(x/128.0*(1.0/10.0-1.0) + 1.0) - 1.0) + 1.0
forkK(x) = 1/(x/128.0*(1.0/(x/128.0*(10.0-1.0) + 1.0) - 1.0) + 1)

Ef = 1.0
Em = 10.0
Vf = 0.09
Vm = 0.49

Mf = Ef / (2.0 * (1.0 + Vf))
Mm = Em / (2.0 * (1.0 + Vm))

Kf = Ef * (Vf - Vf**2) / (1.0 - 3.0 * Vf**2 - 2.0 * Vf**3)

E11(c) = (4.0 * c * (1.0 - c) * (Vf - Vm)**2 * Mm) / ((1.0 - c) * Mm / ())  

WEIGHT = 2

set term png enhanced size 1280,1024#1280, 1024

set output "Mxz_i_10 квадрат.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.4 : 4.5 ]
set xtics 0,0.1,1
set ytics 0.4,0.1,4.5
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Отношение площади включения к общей площади {S_B/S}" font "Monospace, 25"
set ylabel "Коэффициент сдвига {M_{xz}}" font "Monospace, 25" offset 2

plot \
 "mata-quadrate.gpd" using ($1/16384):($12) w l ti "{/Monospace=20 квадрат}" lt 1 lw WEIGHT, \
"/home/primat/projects/tests/heat_test/sources/mata-quadrate.gpd" \
using ($1/16384):(forkK($3)*0.4) w l ti "{/Monospace=20 верхняя граница}" lt 2 lw WEIGHT, \
"/home/primat/projects/tests/heat_test/sources/mata-quadrate.gpd" \
using ($1/16384):(forkS($3)*0.4) w l ti "{/Monospace=20 нижняя граница}" lt 3 lw WEIGHT, \
"/home/primat/projects/tests/elastic_test_on_cell/sources/mata-quadrate.gpd" \
using ($1/16384):($14) w l ti "{/Monospace=20 eee}" lt 4 lw WEIGHT#, \
#"/home/primat/projects/tests/heat_test/sources/mata-quadrate.gpd" \
#using ($1/16384):($2) w l ti "{/Monospace=20 по статье 2}" lt 3 lw WEIGHT, \
#"/home/primat/projects/tests/heat_test/sources/mata-quadrate.gpd" \
#using ($1/16384):(mean($3,serial($3,4.0,0.4),0.4)) w lp ti "{/Monospace=20 My}" lt 6 lw WEIGHT#, \
#"/home/primat/projects/tests/heat_test/sources/mata-quadrate.gpd" \
#using ($1/16384):(serial($1,mean($1,4.0,0.4),0.4)) w lp ti "{/Monospace=20 My}" lt 6 lw WEIGHT#, \
#,\
# "/home/primat/projects/tests/heat_test/sources/mata-quadrate i 10.gpd" \
#using ($1/16384):($2) w l ti "{/Monospace=20 по статье}" lt 2 lw WEIGHT, \

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

set titl "Относительное отклонение" font "Monospace, 20"
set origin 0.14,0.43
set size 0.4, 0.4
set noxlabel 
set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 0.2 ]
set xtics 0,0.1,1
set ytics 0,0.02,0.2
set grid x y
plot \
"mata-quadrate.gpd" using ($1/16384):((forkK($1**0.5)*0.4-$12)/$12) \
w l ti "{/Monospace=20 верхняя граница}" lt 2 lw WEIGHT, \
"mata-quadrate.gpd" using ($1/16384):(($12-forkS($1**0.5)*0.4)/$12) \
w l ti "{/Monospace=20 нижняя граница}" lt 3 lw WEIGHT, \
"mata-quadrate.gpd" using ($1/16384):(((((forkK($1**0.5)+forkS($1**0.5))*0.4/2.0-$12)**2.0)**0.5)/$12) \
w l ti "{/Monospace=20 среднее}" lt 4 lw WEIGHT







