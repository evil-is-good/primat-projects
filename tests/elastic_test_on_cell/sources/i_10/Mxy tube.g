reset

mean(size,l1,l2) = ((size*l1+(128.0-size)*l2)/128.0)
serial(size,l1,l2) = 128.0*l1*l2/(size*l2+(128.0-size)*l1)
derak(x)  = x == 6400?x:-1000000
forkS(x) = x/128.0*(1.0/(x/128.0*(1.0/10.0-1.0) + 1.0) - 1.0) + 1.0
forkK(x) = 1/(x/128.0*(1.0/(x/128.0*(10.0-1.0) + 1.0) - 1.0) + 1)

WEIGHT = 2

set term png enhanced size 1280,1024#1280, 1024

set output "Mxz_i_10 трубка.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.4 : 4.0 ]
set xtics 0,0.1,1
set ytics 0.4,0.1,4.0
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Отношение площади включения к общей площади {S_B/S}" font "Monospace, 25"
set ylabel "Коэффициент сдвига {M_{xz}}" font "Monospace, 25" offset 2

plot \
 "../mata-shell.gpd" using ($1/16384):($12) w l ti "{/Monospace=20 трубка}" lt 1 lw WEIGHT, \
 "../mata-shell.gpd" using ($1/16384):($13) w l ti "{/Monospace=20 верхняя граница}" lt 2 lw WEIGHT, \
 "../mata-shell.gpd" using ($1/16384):($14) w l ti "{/Monospace=20 нижняя граница}" lt 3 lw WEIGHT, \
 "../mata-shell.gpd" using ($1/16384):($12-($13+$14)/2.0) w l ti "{/Monospace=20 нижняя граница}" lt 4 lw WEIGHT#, \

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
"../mata-shell.gpd" using ($1/16384):(($13-$12)/$12) \
w l ti "{/Monospace=20 верхняя граница}" lt 2 lw WEIGHT, \
"../mata-shell.gpd" using ($1/16384):(($12-$14)/$12) \
w l ti "{/Monospace=20 нижняя граница}" lt 3 lw WEIGHT, \
"../mata-shell.gpd" using ($1/16384):(((($12-($13+$14)/2.0)**2.0)**0.5)/$12) \
 w l ti "{/Monospace=20 среднее}" lt 4 lw WEIGHT#, \








