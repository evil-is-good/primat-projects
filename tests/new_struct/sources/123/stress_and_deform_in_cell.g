reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_100_vx_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{xx}^{v_x k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:24-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:24 w l ti "deform" lt 2 lw 3
 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_100_vx_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{zz}^{v_x k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:26-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:26 w l ti "deform" lt 2 lw 3

 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_001_vx_xz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{xz}^{v_x k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:28-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:28 w l ti "deform" lt 2 lw 3

 
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_100_vz_xz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{xz}^{v_z k=(1,0,0)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:31-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:31 w l ti "deform" lt 2 lw 3
  
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_001_vz_xx.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{xx}^{v_z k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:33-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:33 w l ti "deform" lt 2 lw 3
   
 
reset

set term png enhanced font "Monospace,40" size 1620,1024

set output "../res/stress_and_deform_cell_001_vz_zz.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 0.01 ]
#set zrange [ 0.01 : 1 ]
set xtics ("0.0" 0.0, "0.1" 0.001, "0.2" 0.002, "0.3" 0.003, "0.4" 0.004, "0.5" 0.005, "0.6" 0.006, "0.7" 0.007, "0.8" 0.008, "0.9" 0.009, "1.0" 0.01)
#set ytics 0.0,2.0,18
#set key spacing 2.4
#unset key
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "{/Symbol x}_X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol g}_{zz}^{v_z k=(0,0,1)}}" font "Monospace, 25" offset 2

plot \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:35-19 w l ti "stress" lt 1 lw 3, \
 "../hole_plas_cell_E_10_R_25_4_E_10_R_25_4_8.gpd" u 1:35 w l ti "deform" lt 2 lw 3
 
 
 
  

