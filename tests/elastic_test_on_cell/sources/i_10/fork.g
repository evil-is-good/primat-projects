reset

WEIGHT = 2
DEVIDER = 10

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Arial,12" dashed size 800, 600

set output "fork.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.0 : 0.1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.0,0.01,0.1
set key spacing 2.4
set key top left
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {с}" font "Arial, 30"
set ylabel "" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 lw 2 lc rgb "black"

plot \
 "concrete/mata-quadrate.gpd" using ($1/16384):(abs($12-($13+$14)/2.0)/$12) w l ls 1 ti "{/Arial=20 квадрат}", \
 "concrete/mata-cross.gpd"    using ($1/16384):(abs($12-($13+$14)/2.0)/$12) w l ls 2 ti "{/Arial=20 крестовина}"





