reset

mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 2

set terminal png enhanced font "Arial,12" dashed size 800, 600

set output "Mxy_c_h_.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.00 : 0.8 ]
set xtics 0, 0.1, 1#0,1024,16384
set ytics 0.00, 0.1, 0.8
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
      "mata-circ_for_hex.gpd" u ($1/16384):($13-$11) w l ls 1 ti "{тетра}", \
      "mata-hexagon_for_cir.gpd" u ($1):($13-$11) w l ls 2 ti "{гекса}", \
      "mata-quadrate_0.25_turn.gpd" u ($1/16384):(($15-$11)/1.0) w l ls 3 ti "{квадрат}", \
      "mata-quadrate_0.25_turn_xy.gpd" u ($1/16384):($15) w l ls 4 ti "{квадрат}", \
      "mata-quadrate_0.25_unphys_turn.gpd" u ($1/16384):(($3-0.4)/1.0) w l ls 4 ti "{квадрат}"
 

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение от среднего \n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с} " font "Monospace, 20"
set origin 0.54,0.18
set size 0.4, 0.4 #χρόνος ὥρα τέχνη διαχείρισης ἀναδασμός диахора диахерисис ора дихория навория
set noxlabel 
#set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 10.0 ]
#set xtics 0,0.1,1
set ytics 0, 1.0, 10.0
#set nokey
set key bottom right
#set grid x y
#plot "hex_and_circ.gpd" using ($1):($2) w l ls 1 ti "{}"
plot \
"../../mata-quadrate.gpd" u ($1/16384):(($3)/$15) w l ls 1 ti "{квадрат}", \
"../../mata-circ.gpd" u ($1/16384):(($3)/$13) w l ls 2 ti "{круг}", \
"../../mata-cross.gpd" u ($1/16384):(($3)/$26) w l ls 3 ti "{крестовина}", \
"../../mata-shell.gpd" u ($1/16384):(($3)/$15) w l ls 4 ti "{круг}"#, \
#"mata-circ_for_hex.gpd" u ($1/16384):($13-$11)/$11 w l ls 1 ti "{тетра}", \
#"mata-hexagon_for_cir.gpd" u ($1):($13-$11)/$11 w l ls 2 ti "{гекса}", \
#"mata-quadrate.gpd" u ($1/16384):($15-$11)/$11 w l ls 3 ti "{квадрат}"
#"mata-quadrate_0.25_unphys_turn.gpd" u ($1/16384):(($3)/$15) w l ls 1 ti "{тетра}", \





