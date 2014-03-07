reset

load "./common.g"

set output "Mxz.png"
set ylabel "Модуль сдвига G_{xz}, МПа" font "Serif, 30"

set yrange [14 : 38]
set ytics 14, 8, 38

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($12/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($12/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"
