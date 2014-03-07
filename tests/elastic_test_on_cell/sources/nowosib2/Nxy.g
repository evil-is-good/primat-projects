reset

load "./common.g"

set output "Nxy.png"
set ylabel "Коэффициент Пуассона {/Symbol n_{/Nosymbol xy}}" font "Serif, 25"# offset 2

set yrange [0.22 : 0.46]
set ytics 0.22, 0.08, 0.46

set key center right

plot \
 "mata-quadrate2.gpd" using ($1):($3/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($3/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}"
