reset

load "./common.g"

set output "Mxy.png"
set ylabel "Модуль сдвига G_{xy}, МПа" font "Serif, 30"

set yrange [14 : 30]
set ytics 14, 4, 30

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($11/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($11/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"




