reset

set term png enhanced font "Monospace,10" size 1024,720

set output "max_stress_yy_young.png"
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 1 : 100 ]
#set yrange [ 0.01 : 1 ]
set xtics ("0.049" 0.125, "0.196" 0.25, "0.442" 0.375, "0.785" 0.5)
set grid xtics
#set xtics 1,10,101
set border 3
set xtics nomirror
set ytics nomirror
#set ytics 0.0,1.0,18
#set key spacing 2.4
#unset key
set key right center
 set key spacing 1.5 
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Концентрация" font "Monospace, 20"
set ylabel "{/Serif=20 {/Symbol s}_{yy}/ ~{}{0.5⁓}{/Symbol s}_{yy}} " font "Monospace, 25" offset 2

plot \
 "max_1/2/1.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 1}" lt 1 lw 3, \
 "max_1/2/2.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 2}" lt 2 lw 3, \
 "max_1/2/5.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 5}" lt 3 lw 3, \
 "max_1/2/10.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 10}" lt 4 lw 3, \
 "max_1/2/20.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 20}" lt 5 lw 3, \
 "max_1/2/50.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 50}" lt 6 lw 3, \
 "max_1/2/100.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Модуль Юнга волокна = 100}" lt 7 lw 3







