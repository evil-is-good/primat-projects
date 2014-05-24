reset

load "./common.g"

set output "Кнритерий неизотропности.png"
set ylabel "Кнритерий неизотропности" font "Serif, 30"

#set yrange [0 : 0.22]
#set ytics 0, 0.02, 0.22

set key top left

plot \
 "circ.gpd" using ($1):($3) w l ls 1 ti "{/Serif=20 круг}", \
 "cube.gpd" using ($1):($3) w l ls 2 ti "{/Serif=20 квадрат}"
