reset

load "./common.g"

set output "Nzx.png"
set ylabel "Коэффициент Пуассона {/Symbol n_{/Nosymbol zx}}" font "Serif, 25"

set yrange [0.20 : 0.35]
set ytics 0.20, 0.05, 0.35


plot \
 "mata-quadrate2.gpd" using ($1):($4/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($4/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}", \
 "arith.gpd" using ($1):($4/DEVIDER) w l ls 3 ti "{/Serif=20 арифметическое}", \
 "harm.gpd" using ($1):($4/DEVIDER) w l ls 4 ti "{/Serif=20 гармоническое}"
