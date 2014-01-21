reset
load "./common.g"
mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 2

set terminal png enhanced font "Serif,12" dashed dl 5 size 800, 600

set output name."Nxz.png"
set multiplot
set origin 0.01, 0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.2 : 0.3 ]
set xtics 0, 0.1, 1#0,1024,16384
set ytics 0.2 ,0.01, 0.3
set key spacing 2.8
set key bottom right
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Коэффициент армирования {/Symbol \161^{/Nosymbol I}}" font "Serif, 30"
set ylabel "Коэффициент Пуассона {/Symbol n_{/Nosymbol xz}}" font "Serif, 25"# offset 2

set style line 1 lt 1 pt 1 ps 1.5 lw 2 lc rgb "#000000"
set style line 2 lt 1 pt 2 ps 1.5 lw 2 lc rgb "#000000" 
set style line 3 lt 1 pt 3 ps 1.5 lw 2 lc rgb "#000000" 
set style line 4 lt 1 pt 6 ps 1.5 lw 2 lc rgb "#000000" 
set style line 5 lt 1 pt 5 ps 1.5 lw 2 lc rgb "#000000" 
set style line 6 lt 1 pt 7 ps 1.5 lw 2 lc rgb "#000000" 
set style line 7 lt 1 pt 8 ps 2 lw 2 lc rgb "#000000" 

plot \
 dir."mata-quadrate.gpd" using ($1/16384):($8) w lp ls 1 ti "{/Serif=20 квадрат}",\
 dir."mata-shell.gpd" using ($1/16384):($8) w lp ls 2 ti "{/Serif=20 трубка}",\
 dir."mata-cross.gpd" using ($1/16384):($8) w lp ls 3 ti "{/Serif=20 крестовина}", \
 dir."mata-circ.gpd" using ($1/16384):($8) w lp ls 4 ti "{/Serif=20 круг}", \
 "ch_fork.gpd" using ($1/16384):($8) w lp ls 5 ti "{/Serif=20 нижня оценка}" , \
 "ch_fork.gpd" using ($1/16384):($20) w lp ls 6 ti "{/Serif=20 верхняя оценка}", \
 "ch.gpd" using ($1/16384):($8) w lp ls 7 ti "{/Serif=20 формула Хашина-Хилла}"

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение \n формулы Хашина-Хилла от численных \n расчетов \
#{({/Symbol n}_{xz}^{чис}-{/Symbol n}_{xz}^{ХХ})/{/Symbol n}_{xz}^{чис}}" font "Serif, 15"
#set origin 0.16,0.64
#set size 0.4, 0.35
#set noxlabel 
#set xlabel "{S_B/S}" 
#set noylabel
#set xrange [ 0 : 1 ]
#set yrange [ 0 : 0.06 ]
#set xtics 0,0.1,1
#set ytics 0,0.02,1
#set grid x y
#plot \
# "../comparison_q_ch.gpd" u ($1):($8) w lp ls 1 ti "" , \
# "../comparison_s_ch.gpd" u ($1):($8) w lp ls 2 ti "" , \
# "../comparison_cr_ch.gpd" u ($1):($8) w lp ls 3 ti "" , \
# "../comparison_ci_ch.gpd" u ($1):($8) w lp ls 4 ti ""# , \





