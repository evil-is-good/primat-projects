reset

mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 2

set terminal png enhanced font "Arial,12" dashed size 1280, 1024

set output "rand.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.0 : 1 ]
set xtics 0, 0.1, 1#0,1024,16384
set ytics 0.0, 0.1, 1
set key spacing 3.2
set key top center
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
#set xlabel "Коэффициент армирования {/Symbol \161}" font "Arial, 30"
#set ylabel "Коэффициент сдвига {/Symbol m_{/Nosymbol xy}}" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 lw 2 lc rgb "black" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "black" 
set style line 4 lt 4 pt 1 lw 2 lc rgb "black" 

#print rand(0), " ", rand(0)

plot "2.gpd" w p

#plot \
# "../concrete/mata-quadrate.gpd" using ($1/16384):($11) w l ls 1 ti "{/Arial=25 1}"







