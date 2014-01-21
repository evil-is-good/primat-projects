reset

#load "./common.g"

WEIGHT = 2
DEVIDER = 10

file(x) = sprintf("../concrete/%s",x)

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Serif,12" size 800, 600

set output "Ex t.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 7 ]
set yrange [ 0 : 45 ]
set xtics 0, 1, 7
set ytics 0, 5, 45
set key spacing 2.5
set key bottom right
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Возраст бетона в неделях" font "Serif, 30"
set ylabel "Модуль Юнга {E_x}, {ГПа}" font "Serif, 30"# offset 2

set style line 1  lt 1 pt 1 lw 2 lc rgb "#000000"
set style line 2  lt 2 pt 2 lw 2 lc rgb "#000000" 
set style line 3  lt 3 pt 3 lw 2 lc rgb "#000000" 
set style line 4  lt 4 pt 4 lw 2 lc rgb "#000000" 
set style line 5  lt 1 pt 5 lw 2 lc rgb "#000000"
set style line 6  lt 2 pt 6 lw 2 lc rgb "#000000" 
set style line 7  lt 3 pt 7 lw 2 lc rgb "#000000" 
set style line 8  lt 4 pt 8 lw 2 lc rgb "#000000" 
set style line 9  lt 1 pt 9 lw 2 lc rgb "#000000"
set style line 10 lt 2 pt 10 lw 2 lc rgb "#000000" 
set style line 11 lt 3 pt 1 lw 2 lc rgb "#000000" 
set style line 12 lt 4 pt 1 lw 2 lc rgb "#000000" 

plot \
 "mata-circ-0.gpd"  using ($15):($2) w lp ls 1 ti "{/Serif=15 0.0 }", \
 "mata-circ-97.gpd"  using ($15):($2) w lp ls 3 ti "{/Serif=15 0.097 }", \
 "mata-circ-152.gpd"  using ($15):($2) w lp ls 4 ti "{/Serif=15 0.152 }", \
 "mata-circ-219.gpd"  using ($15):($2) w lp ls 5 ti "{/Serif=15 0.219 }", \
 "mata-circ-299.gpd" using ($15):($2) w lp ls 6 ti "{/Serif=15 0.299 }", \
 "mata-circ-390.gpd" using ($15):($2) w lp ls 7 ti "{/Serif=15 0.390 }", \
 "mata-circ-494.gpd" using ($15):($2) w lp ls 8 ti "{/Serif=15 0.494 }", \
 "mata-circ-610.gpd" using ($15):($2) w lp ls 2 ti "{/Serif=15 0.610 }", \
 "mata-circ-738.gpd" using ($15):($2) w lp ls 10 ti "{/Serif=15 0.738 }"#, \
 #"mata-circ-610.gpd" using ($15):((39.88/0.736)*(1-1/(($15*7.0)**0.4))) w l ti ""
 #"mata-circ-39.gpd" using ($15):($2) w l ls 8 ti "{/Serif=15 круг }", \
 #"mata-circ-49.gpd" using ($15):($2) w l ls 9 ti "{/Serif=15 круг }", \
 #"mata-circ-61.gpd" using ($15):($2) w l ls 10 ti "{/Serif=15 круг }", \
 #"mata-circ-73.gpd" using ($15):($2) w l ls 11 ti "{/Serif=15 круг }"#, \
#"mata-circ-24.gpd"  using ($15):($2) w lp ls 2 ti "{/Serif=15 0.024 }", \
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
# "comparison_q_ch.gpd" u ($1):($2) w l ls 1 ti "" , \
# "comparison_s_ch.gpd" u ($1):($2) w l ls 2 ti "" , \
# "comparison_cr_ch.gpd" u ($1):($2) w l ls 3 ti "" , \
# "comparison_ci_ch.gpd" u ($1):($2) w l ls 4 ti ""# , \






