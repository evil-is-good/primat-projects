reset

set terminal png enhanced font "Serif,12" size 800, 600

set output "напряжения-деформации.png"
set origin 0.01, 0.01
set size 0.99, 0.99
#set xrange [ 1.0 : 36.0 ]
#set yrange [ 0 : 45 ]
#set xtics 1.0, 5.0, 36.0
#set ytics 0, 5, 45
set key spacing 2.5
set key bottom right
# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set ylabel "Напрежение" font "Serif, 20"
set xlabel "Деформации" font "Serif, 20"# offset 2

set style line 1  lt 1 pt 1 lw 2 lc rgb "#ff0000"
set style line 2  lt 1 pt 1 lw 2 lc rgb "#0000ff"

plot "max5.gpd" u 3:1 w l ls 1 ti ""






