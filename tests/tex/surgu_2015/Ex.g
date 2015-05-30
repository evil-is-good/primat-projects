reset

load "./common.g"

set output "Ex.png"
set ylabel "{/Serif=30 {/Symbol s}_{yy}/{/Symbol s}_{p}}" font "Serif, 30"

set yrange [0 : 3.1]
set ytics 0, 0.2, 3.1

set key top right

plot \
 "deform_1_yy_line.gpd"  u 1:($2>1.8?$2*1.6:$2) w l ls 1 ti ""
