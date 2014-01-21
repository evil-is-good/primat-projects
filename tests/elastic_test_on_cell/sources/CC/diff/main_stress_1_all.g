reset
set xrange [0:1.0]
set yrange [0:1.0]
set xtics 0, 0.1, 1.0
set ytics 0,0.1,1.0
set size 1.0, 1.0
set term png enhanced size 1024, 1024
unset key
set pm3d map
set xlabel "X"
set ylabel "Y"
set palette function gray > 0.5 ? 1.0 : (gray) * 2.0, gray > 0.5 ? (1.0 - gray) * 2.0: (gray) * 2.0, gray < 0.5 ? 1.0 : 2.0 * (1 - gray)

set title "Stress {/Symbol s}_1 for R = 0.05" font "Serif, 30"
set label "A" at first 0.46875, first 0.460938 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.064" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.421875, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.063" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress5_1.png"
splot "main_stress5.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.1" font "Serif, 30"
set label "A" at first 0.414062, first 0.429688 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.082" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.351562, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.073" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress10_1.png"
splot "main_stress10.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.15" font "Serif, 30"
set label "A" at first 0.375, first 0.398438 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.106" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.28125, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.079" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress15_1.png"
splot "main_stress15.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.2" font "Serif, 30"
set label "A" at first 0.335938, first 0.367188 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.122" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.210938, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.089" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress20_1.png"
splot "main_stress20.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.25" font "Serif, 30"
set label "A" at first 0.296875, first 0.335938 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.131" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.117188, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.102" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress25_1.png"
splot "main_stress25.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.3" font "Serif, 30"
set label "A" at first 0.257812, first 0.304688 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.131" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.0, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.118" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress30_1.png"
splot "main_stress30.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.35" font "Serif, 30"
set label "A" at first 0.195312, first 0.304688 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.125" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.0, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.122" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress35_1.png"
splot "main_stress35.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.4" font "Serif, 30"
set label "A" at first 0.15625, first 0.273438 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.108" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B" at first 0.0, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "B = 0.102" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress40_1.png"
splot "main_stress40.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.45" font "Serif, 30"
set label "A" at first 0.0859375, first 0.289062 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "A = 0.084" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "{/Nosymbol C}	" at first 0.5, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "{/Nosymbol C} = 0.076" at first 0.1, first 0.9 font "Symbol,20" front point pt 7 ps 2 offset 1 
set output "main_stress45_1.png"
splot "main_stress45.gpd" using ($2):($1):($3) with pm3d
set nolabel

set title "Stress {/Symbol s}_1 for R = 0.5" font "Serif, 30"
set label "{/Nosymbol C}" at first 0.5, first 0.5 font "Symbol,20" front point pt 7 ps 2 offset 1 
set label "{/Nosymbol C} = 0.086" at first 0.1, first 0.95 font "Symbol,20" front point pt 7 ps 2 offset 1
set output "main_stress50_1.png"
splot "main_stress50.gpd" using ($2):($1):($3) with pm3d
set nolabel

