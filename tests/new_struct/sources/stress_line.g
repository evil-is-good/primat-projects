reset

set size 1.0, 1.0
set term png enhanced size 1024, 1024
unset key

set origin 0.05, 0.0
set size 0.95, 0.95


set style line 1 lt 1 pt 1 lw 3 lc rgb "black"
set style line 2 lt 2 pt 1 lw 3 lc rgb "black"

set xlabel "r" font "Monospace, 40"

set xtics font "Monospace, 40" out
set ytics font "Monospace, 40"



#set xtics ("20" 0.0, "20.25" 0.25, "20.5" 0.5, "20.75" 0.75, "21" 0.99)"20.5" 0.5, "22.5" 0.5
set xtics ("20" 0.0, "20.25" 0.25, "20.75" 0.75, "21" 0.99) offset 0,-0.7

set ytics 0,1,3
set output "stress_line_zz_H.png"
plot "H/stress_line_zz_2.gpd" w l ls 1

set yrange [0:800]
set ytics 0,200,800
set output"stress_line_yy_H.png"
plot "H/stress_line_yy_1.gpd" w l ls 1


set xtics ("20" 0.0, "21.25" 0.25, "23.25" 0.75, "25" 0.99) offset 0,-0.7

set yrange [0:900]
set ytics 0,300,900
#set ylabel "{/Symbol s_{/Nosymbol θθ}}" font "Serif, 30"
set output "stress_line_yy_W.png"
plot "W/stress_line_yy_1.gpd" w l ls 1

set yrange [-4:10]
set ytics -2,2,8
#set ylabel "{/Symbol s_{/Nosymbol zz}}" font "Serif, 30"
set output "stress_line_zz_W.png"
plot "W/stress_line_zz_2.gpd" w l ls 1






