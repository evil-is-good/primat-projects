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
set palette function gray > 0.5 ? (((gray-0.5)*8.0)**0.5)/4.0 : 0.0, gray > 0.5 ? (gray-0.5)*2.0 : 0.0 , gray > 0.5 ? gray*2.0 : gray*2.0

set title "Stress {/Symbol s}_2 for R = 0.05" font "Serif, 30"
set output "main_stress5_2.png"
splot "main_stress5.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.1" font "Serif, 30"
set output "main_stress10_2.png"
splot "main_stress10.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.15" font "Serif, 30"
set output "main_stress15_2.png"
splot "main_stress15.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.2" font "Serif, 30"
set output "main_stress20_2.png"
splot "main_stress20.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.25" font "Serif, 30"
set output "main_stress25_2.png"
splot "main_stress25.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.3" font "Serif, 30"
set output "main_stress30_2.png"
splot "main_stress30.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.35" font "Serif, 30"
set output "main_stress35_2.png"
splot "main_stress35.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.4" font "Serif, 30"
set output "main_stress40_2.png"
splot "main_stress40.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.45" font "Serif, 30"
set output "main_stress45_2.png"
splot "main_stress45.gpd" using ($2):($1):($4) with pm3d

set title "Stress {/Symbol s}_2 for R = 0.5" font "Serif, 30"
set output "main_stress50_2.png"
splot "main_stress50.gpd" using ($2):($1):($4) with pm3d

