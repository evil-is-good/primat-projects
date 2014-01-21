reset

load "./common.g"

WEIGHT = 2
DEVIDER = 10

file(x) = sprintf("../concrete/%s",x)

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Serif,12" dashed dl 5 size 800, 600

set output name."Ex.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.1 : 1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.1,0.1,1
set key spacing 2.8
set key top left
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {/Symbol \161^{/Nosymbol I}}" font "Serif, 30"
set ylabel "Модуль Юнга {E_x}" font "Serif, 30"# offset 2

set style line 1 lt 1 pt 5 ps 1.5 lw 2 lc rgb "#000000"
set style line 2 lt 1 pt 1 ps 1.5 lw 2 lc rgb "#000000" 
set style line 3 lt 1 pt 4 ps 1.5 lw 2 lc rgb "#000000" 
set style line 4 lt 1 pt 7 ps 1.5 lw 2 lc rgb "#000000" 
set style line 5 lt 1 pt 2 ps 1.5 lw 2 lc rgb "#000000" 
set style line 6 lt 1 pt 3 ps 1.5 lw 2 lc rgb "#000000" 
set style line 7 lt 1 pt 8 ps 2 lw 2 lc rgb "#000000" 

plot \
 dir."mata-quadrate.gpd" using ($1/16384):($2/DEVIDER) w lp ls 1 ti "{/Serif=20 квадрат}", \
 dir."mata-shell.gpd"    using ($1/16384):($2/DEVIDER) w lp ls 2 ti "{/Serif=20 трубка }", \
 dir."mata-cross.gpd"    using ($1/16384):($2/DEVIDER) w lp ls 3 ti "{/Serif=20 крестовина }", \
 dir."mata-circ.gpd"     using ($1/16384):($2/DEVIDER) w lp ls 4 ti "{/Serif=20 круг }", \
 "ch_fork.gpd" using ($1/16384):($2/DEVIDER) w lp ls 5 ti "{/Serif=20 нижня оценка}" , \
 "ch_fork.gpd" using ($1/16384):($14/DEVIDER) w lp ls 6 ti "{/Serif=20 верхняя оценка}", \
 "ch.gpd" using ($1/16384):($2/DEVIDER) w lp ls 7 ti "{/Serif=20 формула Хашина-Хилла}"#, \
 #"../concrete/mata-quadrate.gpd" using ($1/16384):(mean(1,16384-$1)/DEVIDER) w lp ls 7 ti "{/Arial=20 среднее арифметическое" , \
 #"../concrete/mata-quadrate.gpd" using ($1/16384):(serial(16384-$1)/DEVIDER) w lp ls 8 ti "{/Arial=20 среднее гармоническое}"#, \
 #"ch.gpd" using ($1):($2/DEVIDER) w lp ti "{/Monospace=20 7 Кристенсен }" lt 7 lw WEIGHT

#квадра трубка крестовина круг среднее арифметическое среднее гармоническое

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение \n формулы Хашина-Хилла от численных \n расчетов \
#{(E_x^{чис}-E_x^{ХХ})/E_x^{чис}}" font "Serif, 15"
#\n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с}  
#set origin 0.12,0.43
#set size 0.5, 0.55
#set noxlabel 
#set xlabel "{S_B/S}" 
#set noylabel
#set xrange [ 0 : 1 ]
#set yrange [ 0 : 0.3 ]
#set xtics 0,0.1,1
#set ytics 0,0.1,1
#set grid x y
#plot \
# "../comparison_q_ch.gpd" u ($1):($2) w lp ls 1 ti "" , \
# "../comparison_s_ch.gpd" u ($1):($2) w lp ls 2 ti "" , \
# "../comparison_cr_ch.gpd" u ($1):($2) w lp ls 3 ti "" , \
# "../comparison_ci_ch.gpd" u ($1):($2) w lp ls 4 ti ""# , \






