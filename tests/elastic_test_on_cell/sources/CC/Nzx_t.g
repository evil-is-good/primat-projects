reset

#load "./common.g"

WEIGHT = 2
DEVIDER = 10

file(x) = sprintf("../concrete/%s",x)

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Serif,12" size 800, 600

set output "Nzx_t.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 7 ]
set yrange [ 0.1 : 0.212 ]
set xtics 0, 1, 7
set ytics 0.1, 0.01, 0.212
set key spacing 2.5
set key bottom right
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Неделя" font "Serif, 30"
set ylabel "Коэффициент Пуассона {N_{xy}}, {GPa}" font "Serif, 20"# offset 2

set style line 1  lt 1 pt 1 lw 1 lc rgb "#000000"
set style line 2  lt 2 pt 2 lw 1 lc rgb "#000000" 
set style line 3  lt 3 pt 3 lw 1 lc rgb "#000000" 
set style line 4  lt 4 pt 4 lw 1 lc rgb "#000000" 
set style line 5  lt 1 pt 5 lw 1 lc rgb "#000000"
set style line 6  lt 2 pt 6 lw 1 lc rgb "#000000" 
set style line 7  lt 3 pt 7 lw 1 lc rgb "#000000" 
set style line 8  lt 4 pt 8 lw 1 lc rgb "#000000" 
set style line 9  lt 1 pt 9 lw 1 lc rgb "#000000"
set style line 10 lt 2 pt 10 lw 1 lc rgb "#000000" 
set style line 11 lt 3 pt 1 lw 1 lc rgb "#000000" 
set style line 12 lt 4 pt 1 lw 1 lc rgb "#000000" 

plot \
 "mata-circ-0.gpd"  using ($15):($4) w lp ls 1 ti "{/Serif=15 0.0 }", \
 "mata-circ-24.gpd"  using ($15):($4) w lp ls 2 ti "{/Serif=15 0.024 }", \
 "mata-circ-97.gpd"  using ($15):($4) w lp ls 3 ti "{/Serif=15 0.097 }", \
 "mata-circ-152.gpd"  using ($15):($4) w lp ls 4 ti "{/Serif=15 0.152 }", \
 "mata-circ-219.gpd"  using ($15):($4) w lp ls 5 ti "{/Serif=15 0.219 }", \
 "mata-circ-299.gpd" using ($15):($4) w lp ls 6 ti "{/Serif=15 0.299 }", \
 "mata-circ-390.gpd" using ($15):($4) w lp ls 7 ti "{/Serif=15 0.390 }", \
 "mata-circ-494.gpd" using ($15):($4) w lp ls 8 ti "{/Serif=15 0.494 }", \
 "mata-circ-610.gpd" using ($15):($4) w lp ls 9 ti "{/Serif=15 0.610 }", \
 "mata-circ-738.gpd" using ($15):($4) w lp ls 10 ti "{/Serif=15 0.738 }"#, \




