reset

WEIGHT = 2
DEVIDER = 10

mean(x,y) = ((x*y+DEVIDER*(16384-y))/16384)
serial(x) = 16384*DEVIDER/(DEVIDER*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

set terminal png enhanced font "Arial,12" dashed dl 5 size 800, 600

set output "fork.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.0 : 0.1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.0,0.01,0.1
set key spacing 3.2
set key top center
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {/Symbol \161}" font "Arial, 30"
set ylabel "" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 lw 2 lc rgb "black"
#set style line 3 lt 3 pt 1 lw 2 lc rgb "black"
#set style line 4 lt 4 pt 1 lw 2 lc rgb "red"  
#set style line 5 lt 4 pt 1 lw 2 lc rgb "blue"
#set style line 5 lt 4 pt 1 lw 2 lc rgb "green"

plot \
 "../concrete/mata-quadrate.gpd" using ($1/16384):(abs($12-($13+$14)/2.0)/$12) w l ls 1 ti "{/Arial=25 1}", \
 "../concrete/mata-cross.gpd"    using ($1/16384):(abs($12-($13+$14)/2.0)/$12) w l ls 2 ti "{/Arial=25 2}"#, \
 #"../concrete/mata-cross.gpd"    using ($1/16384):(abs($12-(($13+$14)/2.0+$14)/2.0)/$12) w l ls 3 ti "{/Arial=25 2}", \
 #"../concrete/mata-cross.gpd"    using ($1/16384):(abs($12- ((($13+$14)/2.0+$14)/2.0 + ($13+$14)/2.0)/2.0)/$12) w l ls 4 ti "{/Arial=25 3}", \
 #"../concrete/mata-cross.gpd"    using ($1/16384):(abs($12- ((($13+$14)/2.0+$14)/2.0 + ((($13+$14)/2.0+$14)/2.0 + ($13+$14)/2.0)/2.0)/2.0)/$12) w l ls 5 ti "{/Arial=25 4}", \
 #"../concrete/mata-cross.gpd"    using ($1/16384):(abs($12- (((($13+$14)/2.0+$14)/2.0 + ($13+$14)/2.0)/2.0 + ((($13+$14)/2.0+$14)/2.0 + ((($13+$14)/2.0+$14)/2.0 + ($13+$14)/2.0)/2.0)/2.0)/2.0)/$12) w l ls 6 ti "{/Arial=25 5}"





