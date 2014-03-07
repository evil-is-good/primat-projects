reset

load "./common.g"

set output "Ex.png"
set ylabel "Модуль Юнга {E_x}, МПа" font "Serif, 30"

set yrange [40 : 115]
set ytics 40, 15, 115

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($2/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($2/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"
