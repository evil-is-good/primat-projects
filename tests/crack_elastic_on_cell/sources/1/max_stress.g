reset

set xrange [7.0 : 13.0]
set yrange [30.0 : 60.0]
set xtics 7.0, 1.0, 13.0
set ytics 30.0, 5.0 ,60.0
set size 1.0, 1.0
set term png enhanced size 600, 600
unset key
set xlabel "{/Symbol s} / {/Symbol s_{/Nosymbol (max tens)}}"
set ylabel "Количество сломанных ячеек"

#set title "Stress {/Symbol s}_{xx} for i = 2" font "Serif, 30"
set output "max_n.png"
plot "max_n.gpd" u 1:2 w l

