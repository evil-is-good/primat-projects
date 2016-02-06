reset

pow(x,d1,d2) = x < 0.5 ? x < 0.25 ? x**d1 : x**d2 : x > 0.75 ? x**d1 : x**d2

f1(x,d1,d2) =  x > 0.5 ? 1.0 : pow((x) * 2.0, d1, d2)
f2(x,d1,d2) =  x > 0.5 ? pow((1.0 - x) * 2.0, d1, d2) : pow((x) * 2.0, d1, d2)
f3(x,d1,d2) =  x < 0.5 ? 1.0 : pow(2.0 * (1 - x), d1, d2)

#set xrange [0:1.0]
#set yrange [0:1.0]
#set xtics 0,0.1,1.0
#set ytics 0,0.1,1.0
set size 1.0, 1.0
set term png enhanced size 1024, 1024
unset key
set pm3d map
set xlabel "X"
set ylabel "Y"

#gray > 0.25 ? gray < 0.75 ? (gray-0.25)*2.0 : 1.0 : 0.0  (((0.5-gray)*8.0)**0.5)/4.0
#set palette rgbformulae 30, 31, 32
#set palette function gray > 0.5 ? gray-0.5 : 0.0, gray > 0.5 ? gray-0.5 : 0.0, gray**0.8
#set palette function 0.0, gray/1.5, 1.0 - (gray/4.0) gray > 0.5 ? (gray - 0.5)/1.0 : 0.0
set palette function gray > 0.5 ? (((gray-0.5)*8.0)**0.5)/4.0 : 0.0, gray > 0.5 ? (gray-0.5)*2.0 : 0.0 , gray > 0.5 ? gray*2.0 : gray*2.0
#set palette function gray, gray, gray
#set palette function gray > 0.5 ? (1.0 - gray)*0.5 : gray*0.5, gray > 0.5 ? (gray-0.5)*2.0 : 0.0 , gray > 0.5 ? gray*2.0 : gray*2.0

#set cbrange [-1.3:1.3]

#set title "Stress {/Symbol s}_{xx} for R = 0.3" font "Serif, 30"
#set output "sigma_xx.png"
#splot "sigma_x.gpd" using ($2):($1):($3) with pm3d

#set title "Stress {/Symbol s}_{2} for R = 0.3" font "Serif, 30"
#set output "main_stress_2.png"
#splot "main_stress.gpd" using ($2):($1):($4) with pm3d

#set palette function 1.0-gray, 1.0-gray, 1.0

#set title "Stress {/Symbol s}_{xy} for R = 0.3" font "Serif, 30"
#set output "sigma_xy.png"
#splot "sigma_y.gpd" using ($2):($1):(abs($3)) with pm3d

D1 = 2.0
D2 = 0.5#f1(gray, D1, D2), f2(gray, D1, D2), f3(gray, D1, D2)

set palette function \
 gray > 0.5 ? 1.0 : (gray) * 2.0, \
 gray > 0.5 ? (1.0 - gray) * 2.0: (gray) * 2.0, \
 gray < 0.5 ? 1.0 : 2.0 * (1 - gray)
 
#set palette function gray > 0.5 ? ((gray-0.5)*2.0)**0.5 : 0, 0, gray < 0.5 ? (2.0*(0.5-gray))**0.5 : 0


set title "Stress {/Symbol s}_{yy} for R = 0.3" font "Serif, 30"
set output "stress_xx.png"
set zrange [-0.22:-0.04]
splot "stress_xx.gpd" using ($1):($2):($3) with pm3d

#set title "Stress {/Symbol s}_{1} for R = 0.3" font "Serif, 30"
#set output "main_stress_1.png"
#splot "main_stress.gpd" using ($2):($1):($3) with pm3d
#unset pm3d

