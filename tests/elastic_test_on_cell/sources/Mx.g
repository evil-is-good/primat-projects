reset

mean(x,y) = ((x*y+100*(16384-y))/16384)
serial(x) = 16384*100/(100*x+1*(16384-x))
derak(x)  = x == 6400?x:-1000000

WEIGHT = 1

set term png enhanced size 1280,1024#1280, 1024

set output "Mx.png"
set multiplot
set origin 0.01,0.01
set size 0.99, 0.99
set xrange [ 0 : 1 ]
set yrange [ 0.01 : 1 ]
set xtics 0,0.1,1#0,1024,16384
set ytics 0.01,0.11,1
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Отношение площади включения к общей площади {S_B/S}" font "Monospace, 25"
set ylabel "Макротеплопроводность {M_x}" font "Monospace, 25" offset 2

plot \
 "mata-quadrate r-4 new.gpd" using ($1/16384):(mean(1,$1)/100) w l ti "{/Monospace=20 1 среднее арифметическое {{/Symbol=20 l}_{См}}}" lt 4 lw WEIGHT,\
 "mata-quadrate r-4 new.gpd" using ($1/16384):(serial($1)/100) w l ti "{/Monospace=20 2 среднее гармоническое {{/Symbol=20 l}_{Ос}}}" lt 5 lw WEIGHT,\
 "../../cell_heat_test/sources/res/T-qadrate-3.gpd" using ($1/16384):($2/100) w l ti "{/Monospace=20 3 квадрат {{/Symbol=20 l}_{Кв}}}" lt 1 lw WEIGHT,\
 "../../cell_heat_test/sources/res/T-shell-3.gpd" using ($1/16384):($2/100) with lines ti "{/Monospace=20 4 трубка {{/Symbol=20 l}_{Тр}}}" lt 2 lw WEIGHT,\
 "../../cell_heat_test/sources/res/T-cross-3.gpd" using ($1/16384):($2/100) with lines ti "{/Monospace=20 5 крестовина {{/Symbol=20 l}_{Кр}}}" lt 3 lw WEIGHT, \
 "/home/primat/deal.ii-data/test3/last/circle resize lin~" using ($1/16384):($2/100) with lines ti "{/Monospace=20 6 круг {{/Symbol=20 l}_{Кр}}}" lt 6 lw WEIGHT

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







