reset

load "./common.g"

set output "Ez.png"
set ylabel "Модуль Юнга {E_z}, МПа" font "Serif, 30"

set yrange [1 : 5]
set ytics 1, 1, 5

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($10/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($10/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}", \
 "arith.gpd" using ($1):($10/DEVIDER) w l ls 3 ti "{/Serif=20 арифметическое}", \
 "harm.gpd" using ($1):($10/DEVIDER) w l ls 4 ti "{/Serif=20 гармоническое}"
