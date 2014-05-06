WEIGHT = 2
DEVIDER = 1

set terminal png enhanced font "Serif,12" dashed dl 5 size 800, 600


set origin 0.01, 0.01
set size 0.99, 0.99
set xrange [0.0 : 1.0]
set xtics 0, 0.2, 1

set xlabel "Коэффициент армирования {/Symbol \161^{/Nosymbol I}}" font "Serif, 30"


#set style line 1 lt 1 pt 5 ps 1.5 lw 2 lc rgb "#000000" 
set style line 1 lt 1 pt 1 lw 2 lc rgb "blue"
set style line 2 lt 2 pt 1 lw 2 lc rgb "red" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "green" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "black" 
set style line 5 lt 1 pt 1 lw 2 lc rgb "black" 
set style line 6 lt 1 pt 2 lw 2 lc rgb "black" 

set key spacing 2.8
