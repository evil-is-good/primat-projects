reset

set term png enhanced font "Monospace,10" size 1024,720

set output "../res/max_stress_zz_25_hole_and_not_hole.png"
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
set key right bottom
 set key spacing 1.5 
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Отношение модуля Юнга волокна к модулю Юнга связующего" font "Monospace, 20"
set ylabel "{/Serif=20 {/Symbol s}_{zz}/ ~{}{0.5⁓}{/Symbol s}_{zz}} " font "Monospace, 25" offset 2

plot \
 "../max_2/max_25.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Образец с круглым отверстием}" lt 1 lw 3, \
 "../max_3/max_25.gpd" u 1:($3/$2) w l ti "{/Monospace=18 Образец без выреза}" lt 2 lw 3







