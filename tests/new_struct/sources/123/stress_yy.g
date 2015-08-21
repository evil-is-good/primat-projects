reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 0 : 1 ]
#set zrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol s}_{zz}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:4 w l ti "" lt 1 lw 3#, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_9.gpd" u 1:4 w l ti "" lt 2 lw 3, \
 "../hole_plas_cell_E_50_R_25_4_E_50_R_25_4_9.gpd" u 1:4 w l ti "" lt 3 lw 3
 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 0 : 1 ]
#set zrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol s}_{xx}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:2 w l ti "" lt 1 lw 3
 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_100_vx_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{xx}^{v_x k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:5 w l ti "" lt 1 lw 3
 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_100_vx_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{zz}^{v_x k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:7 w l ti "" lt 1 lw 3

 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_001_vx_xz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{xz}^{v_x k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:9 w l ti "" lt 1 lw 3

 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_100_vz_xz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{xz}^{v_z k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:12 w l ti "" lt 1 lw 3
  
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_001_vz_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{xx}^{v_z k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:14 w l ti "" lt 1 lw 3
   
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_cell_001_vz_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol t}_{zz}^{v_z k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_4_9.gpd" u 1:16 w l ti "" lt 1 lw 3
 
 
 
  
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/macrostress_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 0 : 1 ]
#set zrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "X" font "Monospace, 50"
set ylabel "{/Serif=50 ~{}{0.5⁓}{/Symbol s}_{xx}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_5_9.gpd" u 1:17 w l ti "" lt 1 lw 3
   
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/macrostress_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 0 : 1 ]
#set zrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
#set ytics 0.0,2.0,18
#set key spacing 2.4
unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "X" font "Monospace, 50"
set ylabel "{/Serif=50 ~{}{0.5⁓}{/Symbol s}_{zz}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_5_9.gpd" u 1:20 w l ti "" lt 1 lw 3
