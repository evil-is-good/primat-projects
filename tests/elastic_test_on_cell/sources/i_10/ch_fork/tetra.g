set xrange [ -3.0 : 3.0 ]
set yrange [ -3.0 : 3.0 ]
set xtics -3.0, 0.2, 3.0
set ytics -3.0, 0.2, 3.0
set parametric 
set nokey
set noxtics
set noytics


set terminal png enhanced font "Arial,12" dashed size 800, 800
set output "tetra.png"

plot cos(t)-3,sin(t)-3 lt 1, cos(t)-3,sin(t) lt 1, cos(t)-3,sin(t)+3 lt 1, \
     cos(t)-0,sin(t)-3 lt 1, cos(t)-0,sin(t) lt 1, cos(t)-0,sin(t)+3 lt 1, \
     cos(t)+3,sin(t)-3 lt 1, cos(t)+3,sin(t) lt 1, cos(t)+3,sin(t)+3 lt 1

set terminal latex
set output "tetra.tex"
plot cos(t)-3,sin(t)-3 lt 1, cos(t)-3,sin(t) lt 1, cos(t)-3,sin(t)+3 lt 1, \
     cos(t)-0,sin(t)-3 lt 1, cos(t)-0,sin(t) lt 1, cos(t)-0,sin(t)+3 lt 1, \
     cos(t)+3,sin(t)-3 lt 1, cos(t)+3,sin(t) lt 1, cos(t)+3,sin(t)+3 lt 1
