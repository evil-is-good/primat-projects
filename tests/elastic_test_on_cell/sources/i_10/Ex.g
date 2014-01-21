reset

WEIGHT = 2
DEVIDER = 10

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Arial,12" dashed size 800, 600

set output "Ex_i_10.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.1 : 1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.1,0.1,1
set key spacing 4.4
set key top left
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {с}" font "Arial, 30"
set ylabel "Модуль Юнга {E_x}" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 lw 2 lc rgb "black" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "black" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "black" 
set style line 5 lt 1 pt 1 lw 2 lc rgb "black" 
set style line 6 lt 1 pt 2 lw 2 lc rgb "black" 

plot \
 "concrete/mata-quadrate.gpd" using ($1/16384):($2/DEVIDER) w l ls 1 ti "{/Arial=30 квадрат}", \
 "concrete/mata-shell.gpd"    using ($1/16384):($2/DEVIDER) w l ls 2 ti "{/Arial=20 трубка }", \
 "concrete/mata-cross.gpd"    using ($1/16384):($2/DEVIDER) w l ls 3 ti "{/Arial=20 крестовина }", \
 "concrete/mata-circ.gpd"     using ($1/16384):($2/DEVIDER) w l ls 4 ti "{/Arial=20 круг }", \
 "concrete/mata-quadrate.gpd" using ($1/16384):(mean(1,16384-$1)/DEVIDER) w lp ls 5 ti "{/Arial=20 среднее арифметическое" , \
 "concrete/mata-quadrate.gpd" using ($1/16384):(serial(16384-$1)/DEVIDER) w lp ls 6 ti "{/Arial=20 среднее гармоническое}"#, \
 #"ch.gpd" using ($1):($2/DEVIDER) w lp ti "{/Monospace=20 7 Кристенсен }" lt 7 lw WEIGHT

#квадра трубка крестовина круг среднее арифметическое среднее гармоническое

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение от среднего \n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с} " font "Monospace, 20"
#set origin 0.54,0.43
#set size 0.4, 0.4
#set noxlabel 
#set xlabel "{S_B/S}" 
#set noylabel
#set xrange [ 0 : 1 ]
#set yrange [ 0 : 1 ]
#set xtics 0,0.1,1
#set ytics 0,0.1,1
#set grid x y
#plot "cube/unconst area/6" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3, "shell/unconst area/7" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3, "plas/unconst area/7" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3







