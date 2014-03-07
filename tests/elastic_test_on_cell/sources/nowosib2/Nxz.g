reset

load "./common.g"

set output "Nxz.png"
set ylabel "Коэффициент Пуассона {/Symbol n_{/Nosymbol xz}}" font "Serif, 25"

set yrange [0.35 : 0.38]
set ytics 0.35, 0.01, 0.38

set key top left

plot \
 "mata-quadrate2.gpd" using ($1):($8/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($8/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"
