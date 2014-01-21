load "num.gpd"

set terminal png enhanced font "Serif,12" size 800, 600

set output "num.png"
set xrange [ 0 : 1.0 ]
set yrange [ 0 : 1.0 ]
set xtics 0, 0.1, 1
set ytics 0, 0.1, 1
set pm3d map

splot "res_xx.gpd" w l
