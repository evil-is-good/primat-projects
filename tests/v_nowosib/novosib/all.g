reset

WEIGHT = 2
DEVIDER = 100

arithmetic(s,l1,l2) = ((s*l2+l1*(1.0-s))/1.0)
harmonic(s,l1,l2) = 1.0 / (s/l2 + (1.0 - s)/l1)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Serif,12" size 800, 600

set output "1.png"
set multiplot
set origin 0.00,0.00
set size 1.0, 1.0
set xrange [ 0 : 1 ]
set yrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.01,0.09,1
set key spacing 2.1
set key top right
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {/Symbol \161^{/Nosymbol I}}" font "Serif, 30"
set ylabel "Коэффициент теплопроводности {{/Symbol l}_{xx}}" font "Serif, 20"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "#000000"
set style line 2 lt 2 pt 1 lw 2 lc rgb "#00FFFF" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "#FF00FF" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "#FF7F00" 
set style line 5 lt 1 pt 5 ps 1.5 lw 2 lc rgb "red" 
set style line 6 lt 1 pt 7 ps 1.5 lw 2 lc rgb "red" 
set style line 7 lt 1 pt 10 lw 2 lc rgb "blue" 

plot \
 "../mata-quadrate.gpd" using ($1):($2/DEVIDER) w l ls 1 ti "{/Serif=15 квадрат}", \
 "../mata-shell.gpd"    using ($1):($2/DEVIDER) w l ls 2 ti "{/Serif=15 трубка }", \
 "../mata-cross.gpd"    using ($1):($2/DEVIDER) w l ls 3 ti "{/Serif=15 крестовина }", \
 "../mata-circ.gpd"     using ($1):($2/DEVIDER) w l ls 4 ti "{/Serif=15 круг }", \
 "../mata-quadrate.gpd" using ($1):(arithmetic($1,100,1)/DEVIDER) w lp ls 5 ti "{/Serif=15 формула Рейсса }", \
 "../mata-quadrate.gpd" using ($1):(harmonic($1,100,1)/DEVIDER) w lp ls 6 ti "{/Serif=15 формула Фойгта }"


#квадра трубка крестовина круг среднее арифметическое среднее гармоническое

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

set titl "Относительное отклонение \n формулы Рейсса от численных \n расчетов \
{({/Symbol l}^{чис}-{/Symbol l}^{Р})/{/Symbol l}^{чис}}" font "Serif, 15"
#\n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с}  
set origin 0.55,0.3
set size 0.37, 0.46
set noxlabel 
#set xlabel "{S_B/S}" 
set noylabel
set xrange [ 0 : 1 ]
set yrange [ 0 : 1.6 ]
set xtics 0,0.2,1
set ytics 0,0.4,1.6
set grid x y
plot \
 "../mata-quadrate.gpd" u ($1):((arithmetic($1,100,1) - $2) / $2) w l ls 1 ti "" , \
 "../mata-shell.gpd" u ($1):((arithmetic($1,100,1) - $2) / $2) w l ls 2 ti "" , \
 "../mata-cross.gpd" u ($1):((arithmetic($1,100,1) - $2) / $2) w l ls 3 ti "" , \
 "../mata-circ.gpd" u ($1):((arithmetic($1,100,1) - $2) / $2) w l ls 4 ti ""# , \






