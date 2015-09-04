reset

a = 0.05

sigma_theta(x) = (1.0/2.0) * (1.0 + 1.0/(x**2.0) + (1.0 + 3.0/(x**4.0))) 
sigma_r(x) = ((x > -0.05) && (x < 0.05)) ? 0.0 : \
((1.0/2.0) * (1.0 - (a**2.0)/(x**2.0) - (1.0 + (3.0*a**4.0)/(x**4.0) - (4.0*a**2.0)/(x**2.0))))

set term png enhanced font "Monospace,10" size 1024,720

set output "vard.png"
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0.0 : 1 ]
#set zrange [ 0.01 : 1 ]~{}{0.0\062}
#set xtics ("1" 1, "2" 2, "5" 5, "10" 10, "20" 20, "50" 50, "100" 100)
#set grid xtics
#set xtics 1,10,101
#set border 3
#set xtics nomirror
#set ytics nomirror
#set ytics 0.0,1.0,18
#set key spacing 2.4
#unset key
set key right center
set key spacing 1.5 
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "X" font "Monospace, 20"
#set ylabel "{/Serif=50 ~{}{0.5⁓}{/Symbol s}_{xx}}" font "Monospace, 10" offset 2

#plot sigma_r(x-0.5) w l ti "{/Symbol s}_{r}" lt 1 lw 3
plot \
 "../hole_plas_cell_E_10_R_25_5_E_10_R_25_5_9.gpd" u 1:(sigma_r($1-0.5)) w l ti "" lt 1 lw 3







