reset

load "./common.g"

set output "Mxy.png"
set ylabel "Модуль сдвига G_{xy}, МПа" font "Serif, 30"

set yrange [0.37 : 1.87]
set ytics 0.37, 0.2, 1.87

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($11/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($11/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"




