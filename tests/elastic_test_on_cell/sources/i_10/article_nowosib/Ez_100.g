reset

load "./common.g"

WEIGHT = 2
DEVIDER = 100

file(x) = sprintf("../concrete/%s",x)

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Serif,12" size 800, 600

set output name."Ez.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.01,0.1,1
set key spacing 2.5
set key bottom right
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {/Symbol \161^{/Nosymbol I}}" font "Serif, 30"
set ylabel "Модуль Юнга {E_z}" font "Serif, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "#000000"
set style line 2 lt 2 pt 1 lw 2 lc rgb "#00FFFF" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "#FF00FF" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "#FF7F00" 
set style line 5 lt 1 pt 5 ps 1.5 lw 2 lc rgb "red" 
set style line 6 lt 1 pt 7 ps 1.5 lw 2 lc rgb "red" 
set style line 7 lt 1 pt 10 lw 2 lc rgb "blue" 

plot \
 dir."mata-quadrate.gpd" using ($1/16384):($10/DEVIDER) w l ls 1 ti "{/Serif=15 квадрат}", \
 dir."mata-shell.gpd"    using ($1/16384):($10/DEVIDER) w l ls 2 ti "{/Serif=15 трубка }", \
 dir."mata-cross.gpd"    using ($1/16384):($10/DEVIDER) w l ls 3 ti "{/Serif=15 крестовина }", \
 dir."mata-circ.gpd"     using ($1/16384):($10/DEVIDER) w l ls 4 ti "{/Serif=15 круг }", \
 "../../ch_fork.gpd" using ($1/16384):($10/DEVIDER) w lp ls 5 ti "{/Serif=15 нижня оценка}" , \
 "../../ch_fork.gpd" using ($1/16384):($22/DEVIDER) w lp ls 6 ti "{/Serif=15 верхняя оценка}", \
 "../../ch.gpd" using ($1/16384):($10/DEVIDER) w l ls 7 ti "{/Serif=15 формула Хашина-Хилла}"#, \
 #"../concrete/mata-quadrate.gpd" using ($1/16384):(mean(1,16384-$1)/DEVIDER) w lp ls 7 ti "{/Arial=20 среднее арифметическое" , \
 #"../concrete/mata-quadrate.gpd" using ($1/16384):(serial(16384-$1)/DEVIDER) w lp ls 8 ti "{/Arial=20 среднее гармоническое}"#, \
 #"ch.gpd" using ($1):($2/DEVIDER) w lp ti "{/Monospace=20 7 Кристенсен }" lt 7 lw WEIGHT

#квадра трубка крестовина круг среднее арифметическое среднее гармоническое

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

set titl "Относительное отклонение \n формулы Хашина-Хилла от численных \n расчетов \
{(E_z^{чис}-E_z^{ХХ})/E_z^{чис}}" font "Serif, 15"
#\n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с}  
set origin 0.11	,0.53
set size 0.45, 0.45
set noxlabel 
#set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 0.01 ]
set xtics 0,0.1,1
set ytics 0,0.002,0.01
set grid x y
plot \
 "comparison_q_ch.gpd" u ($1):($10) w l ls 1 ti "" , \
 "comparison_s_ch.gpd" u ($1):($10) w l ls 2 ti "" , \
 "comparison_cr_ch.gpd" u ($1):($10) w l ls 3 ti "" , \
 "comparison_ci_ch.gpd" u ($1):($10) w l ls 4 ti ""# , \






