load "sh_f.g"

set terminal png enhanced font "Serif,12" size 800, 600

set output "sh_f.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1.0 ]
set yrange [ 0 : 1.5 ]
set xtics 0, 0.1, 1
set ytics 0, 0.1, 1
set key spacing 2.5
set key bottom right
set xlabel "X"
set ylabel "Y"

splot f(x,y), grad_x(y)
