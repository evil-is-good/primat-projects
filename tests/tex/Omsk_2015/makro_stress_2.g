reset

WEIGHT = 8

set term png enhanced font "Monospace,40" size 1620,1024 #1280, 1024

set output "makro_stress_z_z_x.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
#set xrange [ 0 : 1 ]
#set yrange [ 0.0 : 2 ]
set xtics 0,0.1,1#0,1024,16384
#set ytics 0.0,0.2,2
#set key spacing 2.4
unset key
set xlabel "X" font "Monospace, 50"
set ylabel "{/Serif=50 {/Symbol w}_{zzx}" font "Monospace, 25" offset 2

plot \
 "../05_8_25_5/final/hole_plas_cell.gpd" u 1:3 w l ti "{/Monospace=20 1 среднее арифметическое {{/Symbol=20 l}_{См}}}" lt 4 lw 3
