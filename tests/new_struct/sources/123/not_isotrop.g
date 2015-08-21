reset

set term png enhanced font "Monospace,10" size 1024,720

set output "../res/not_isotrop.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 1 : 100 ]
#set zrange [ 0.01 : 1 ]~{}{0.0\062}
set xtics ("1" 1, "2" 2, "5" 5, "10" 10, "20" 20, "50" 50, "100" 100)
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
set xlabel "Отношение модуля Юнга волокна к модулю Юнга связующего" font "Monospace, 20"
set ylabel "{/Serif=20 ({/Symbol m}_{iso} - {/Symbol m}_{xy})/{/Symbol m}_{iso}} " font "Monospace, 25" offset 2

plot \
 "../not_isotrop_125.gpd" u 1:(($2-$5)/$2) w l ti "{/Monospace=18 Концентрация 0.071}" lt 1 lw 3, \
 "../not_isotrop_25.gpd" u 1:(($2-$5)/$2) w l ti "{/Monospace=18 Концентрация 0.196}" lt 2 lw 3, \
 "../not_isotrop_375.gpd" u 1:(($2-$5)/$2) w l ti "{/Monospace=18 Концентрация 0.442}" lt 3 lw 3







