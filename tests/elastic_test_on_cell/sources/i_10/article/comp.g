reset

mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 2

set terminal png enhanced font "Arial,12" dashed dl 5 size 800, 600

set output "Comp.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.0 : 0.6 ]
set xtics 0, 0.1, 1#0,1024,16384
set ytics 0.0, 0.1, 1
set key spacing 3.2
set key top right
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Коэффициент армирования {/Symbol \161}" font "Arial, 30"
set ylabel "" font "Arial, 30"# offset 2

set style line 1 lt 1 pt 1 lw 2 lc rgb "black"
set style line 2 lt 2 pt 1 ps 0.5 lw 2 lc rgb "black" 
set style line 3 lt 3 pt 1 lw 2 lc rgb "black" 

plot \
 "../concrete/comp_c.gpd" using ($1):(($3-$2)/$2) w l ls 1 ti "",\
 "../concrete/comp_cs.gpd" using ($1):(($3-$2)/$2) w l ls 2 ti "",\
 "../concrete/comp_cs.gpd" using ($1):(($4-$2)/$2) w l ls 3 ti ""

#, "all/l" with labels font "Monospace, 20" ti ""#, "plas/unconst area/7" using (derak($1)/16384):($2/100) with points ti "" lt -1 pt 1 ps 3#, "shell/unconst area/7" using (derak($1)/16384):($2/100) with points ti "{/Monospace=20 S_в = 0.39}" lt -1 pt 1 ps 3, "all/l1" with labels font "Monospace, 20" ti ""

#set titl "Относительное отклонение от среднего \n ({/Symbol=20 l_s} - {{/Symbol=20 l}_с}) / {{/Symbol=20 l}_с} " font "Monospace, 20"
#set origin 0.54,0.43
#set size 0.4, 0.4
#set noxlabel 
#set xlabel "{S_B/S}" 
#set noylabel
#set xrange [ 0 : 1 ]
#set yrange [ 0 : 1 ]
#set xtics 0,0.1,1
#set ytics 0,0.1,1
#set grid x y
#plot "cube/unconst area/6" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3, "shell/unconst area/7" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3, "plas/unconst area/7" using ($1/16384):(((mean(1,$1)-$2))/mean(1,$1)) with lines ti "" lw 3







