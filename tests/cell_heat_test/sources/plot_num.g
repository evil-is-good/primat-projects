load "num1.gpd"

set terminal png enhanced font "Serif,12" size 800, 600

set output "num1.png"
set xrange [ 0 : 1.0 ]
set yrange [ 0 : 1.0 ]
set xtics 0, 0.1, 1
set ytics 0, 0.1, 1
set pm3d map

splot "res_x.gpd" w l ti ""
