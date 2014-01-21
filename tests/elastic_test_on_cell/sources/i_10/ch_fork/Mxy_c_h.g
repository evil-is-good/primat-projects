reset

mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 2

set terminal png enhanced font "Arial,12" dashed #size 8, 6

set output "Mxy_c_h.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.00 : 2.0 ]
set xtics 0, 0.1, 1#0,1024,16384
set ytics 0.00, 0.2, 2.0
set key spacing 2.4
set key top left
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Коэффициент армирования {/Symbol \161}" font "Arial, 30"
set ylabel "Коэффициент сдвига {/Symbol m_{/Nosymbol xy}}" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 lw 2 lc rgb "black" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "black" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "black" 
set style line 5 lt 1 pt 1 lw 2 lc rgb "red" 
set style line 6 lt 1 pt 2 lw 2 lc rgb "red" 
set style line 7 lt 1 pt 1 lw 2 lc rgb "red" 
set style line 8 lt 1 pt 2 lw 2 lc rgb "red"

plot \
 "mata-circ_for_hex.gpd" using ($1/16384):($11) w l lt 1 ti "{/Arial=20 тетра действительное}", \
 "mata-hexagon_for_cir.gpd" using ($1):($11) w lp lt 2 ti "{/Arial=20 гекса действительное}", \
 "mata-circ_for_hex.gpd" using ($1/16384):($13) w l lt 3 ti "{/Arial=20 тетра предпологаемое}", \
 "mata-hexagon_for_cir.gpd" using ($1):($13) w lp lt 4 ti "{/Arial=20 гекса предпологаемое}", \
 "mata-quadrate.gpd" using ($1/16384):($11) w l ls 1 ti "{/Arial=20 квадрат действительное}", \
 "mata-quadrate.gpd" using ($1/16384):($15) w l ls 2 ti "{/Arial=20 квадрат предпологаемое}"#, \
 #"../../ch_fork.gpd" using ($1/16384):($11) w lp ls 5 ti "{/Arial=20 нижня оценка}" , \
 #"../../ch_fork.gpd" using ($1/16384):($23) w lp ls 6 ti "{/Arial=20 верхняя оценка}"
 

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение от среднего \n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с} " font "Monospace, 20"
set origin 0.54,0.18
set size 0.4, 0.4
set noxlabel 
#set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 0.5 ]
#set xtics 0,0.1,1
set ytics 0, 0.04, 0.5
#set nokey
set key bottom right
#set grid x y
#plot "hex_and_circ.gpd" using ($1):($2) w l ls 1 ti "{}"
plot "mata-circ_for_hex.gpd" u ($1/16384):($13-$11)/$11 w l ls 1 ti "{тетра}", \
"mata-hexagon_for_cir.gpd" u ($1):($13-$11)/$11 w l ls 2 ti "{гекса}", \
"mata-quadrate.gpd" u ($1/16384):($15-$11)/$11 w l ls 3 ti "{квадрат}"






