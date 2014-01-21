set label "0.064" at first 0.461, first 0.461 font "Symbol,24"

set terminal png enhanced font "Serif,12" size 800, 600

set output "num.png"
set xrange [ 0 : 1.0 ]
set yrange [ 0 : 1.0 ]
set xtics 0, 0.1, 1
set ytics 0, 0.1, 1

splot "main_stress5.gpd" w l
