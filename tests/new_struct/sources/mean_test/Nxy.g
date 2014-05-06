reset

load "./common.g"

set output "Nxy.png"
set ylabel "Коэффициент Пуассона {/Symbol n_{/Nosymbol xy}}" font "Serif, 25"# offset 2

#set yrange [0.24 : 0.40]
#set ytics 0.24, 0.04, 0.4
set yrange [0.20 : 0.35]
set ytics 0.20, 0.05, 0.35

set key top right

plot \
 "mata-quadrate2.gpd" using ($1):($3/DEVIDER) w l ls 1 ti "{/Serif=20 тетрагональная}", \
 "mata-hexagon2.gpd" using ($1):($3/DEVIDER) w l ls 2 ti "{/Serif=20 гексагональная}", \
 "arith.gpd" using ($1):($3/DEVIDER) w l ls 3 ti "{/Serif=20 арифметическое}", \
 "harm.gpd" using ($1):($3/DEVIDER) w l ls 4 ti "{/Serif=20 гармоническое}"
