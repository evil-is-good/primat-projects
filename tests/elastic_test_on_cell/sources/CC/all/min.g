reset

set terminal png enhanced font "Serif,12" size 800, 600

set origin 0.01,0.01
set size 0.99, 0.99

set style line 1 lt 1 pt 1 lw 2 lc rgb "#000000"
set style line 2 lt 2 pt 1 lw 2 lc rgb "#00FFFF" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "#FF00FF" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "#FF7F00" 
set style line 5 lt 1 pt 5 ps 1.5 lw 2 lc rgb "red" 
set style line 6 lt 1 pt 7 ps 1.5 lw 2 lc rgb "red" 
set style line 7 lt 1 pt 10 lw 2 lc rgb "blue" 
set key spacing 2.5
set key bottom right

#set xrange [ 0 : 1 ]
#set yrange [ 0.1 : 1 ]
#set xtics 0,0.1,1
#set ytics 0.1,0.1,1

# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}

set output "min_main_stress_2.png"

#set xrange [ 0 : 1 ]
#set yrange [ 0.1 : 1 ]
#set xtics 0,0.1,1
#set ytics 0.1,0.1,1

# set title "Sine [-Pi..Pi]"
# Отношение площади включения к общей площади {S_B/S}
set xlabel "Коэффициент армирования {/Symbol \161}" font "Serif, 30"
set ylabel "{/Symbol s_2^{/Nosymbol min}}" font "Serif, 30"# offset 2


plot \
 "min-max" using ($1*$1*3.14159265359):($5) w l ls 1 ti "" #"{/Serif=15 квадрат}"





