reset


set xrange [0:1.0]
set yrange [0:1.0]
set xtics 0,0.1,1.0
set ytics 0,0.1,1.0
set size 1.0, 1.0
set term png size 1024, 1024
unset key
set pm3d map
set xlabel "X"
set ylabel "Y"

set palette function \
 gray > 0.5 ? 1.0 : (gray) * 2.0, \
 gray > 0.5 ? (1.0 - gray) * 2.0: (gray) * 2.0, \
 gray < 0.5 ? 1.0 : 2.0 * (1 - gray)

set label "*" at first 0.461, first 0.461 center font "Symbol,24" front
set label " 0.064" at first 0.461, first 0.461 font "Symbol,20" front
set label "*" at first 0.461, first 0.461 center font "Symbol,24" front
set label " 0.064" at first 0.461, first 0.461 font "Symbol,20" front
set label "*" at first 0.461, first 0.461 center font "Symbol,24" front
set label " 0.064" at first 0.461, first 0.461 font "Symbol,20" front

set output "main_stress5.png"
splot "main_stress5.gpd" using ($2):($1):($3) with pm3d

set output "main_stress10.png"
splot "main_stress10.gpd" using ($2):($1):($3) with pm3d

set output "main_stress15.png"
splot "main_stress15.gpd" using ($2):($1):($3) with pm3d

set output "main_stress20.png"
splot "main_stress20.gpd" using ($2):($1):($3) with pm3d

set output "main_stress25.png"
splot "main_stress25.gpd" using ($2):($1):($3) with pm3d

set output "main_stress30.png"
splot "main_stress30.gpd" using ($2):($1):($3) with pm3d

set output "main_stress35.png"
splot "main_stress35.gpd" using ($2):($1):($3) with pm3d

set output "main_stress40.png"
splot "main_stress40.gpd" using ($2):($1):($3) with pm3d

set output "main_stress45.png"
splot "main_stress45.gpd" using ($2):($1):($3) with pm3d

set output "main_stress50.png"
splot "main_stress50.gpd" using ($2):($1):($3) with pm3d
