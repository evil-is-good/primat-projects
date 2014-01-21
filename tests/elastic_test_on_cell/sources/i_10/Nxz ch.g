reset

WEIGHT = 2

set term png enhanced size 1280,1024#1280, 1024

set output "Nxz ch.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.2 : 0.28 ]
set xtics 0,0.1,1
set ytics 0.2, 0.01, 0.28
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Отношение площади включения к общей площади {S_B/S}" font "Monospace, 25"
set ylabel "Коэффициент Пуассона {N_{xz}}" font "Monospace, 25" offset 2

plot \
 "../mata-quadrate.gpd" using ($1/16384):($8) w l ti "{/Monospace=20 квадрат}" lt 1 lw WEIGHT, \
 "../mata-quadrate.gpd" using ($1/16384):($21) w l ti "{/Monospace=20 Кристенсен}" lt 2 lw WEIGHT


set titl "Относительное отклонение" font "Monospace, 20"
set origin 0.14,0.43
set size 0.4, 0.4
set noxlabel 
set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 0.02 ]
set xtics 0,0.1,1
set ytics 0,0.002,0.02
set grid x y
plot \
 "../mata-quadrate.gpd" using ($1/16384):(abs($21-$8)/$21) w l ti "{/Monospace=20 ch}" lt 1 lw WEIGHT







