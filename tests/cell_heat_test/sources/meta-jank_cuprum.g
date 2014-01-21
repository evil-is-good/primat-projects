reset

WEIGHT = 1

set term png enhanced size 1280,1024#1280, 1024

set output "Теплопроводность вдоль оси Ox cuprum.png"
set origin 0.01, 0.01
set size 0.99, 0.99
set xrange [ 0.01 : 0.99 ]
#set yrange [ 0.01 : 0.99 ]
#set xtics 0, 0.1, 1#
#set ytics 0.01 ,0.11 ,1
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "Коэффициент армирования" font "Monospace, 25"
set ylabel "Коэффициент теплопроводности {{/Symbol=20 l}_x}" font "Monospace, 25" offset 2

plot \
 "meta-hex_cuprum.gpd" u 1:2 w l ti "{/Monospace=20 численное решение}" lw 2 lt 1, \
 "jancovsky_cuprum.gpd" u 1:2 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
 "jancovsky_cuprum.gpd" u 1:4 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3
 

set output "Теплопроводность вдоль оси Oy cuprum.png"
set ylabel "Коэффициент теплопроводности {{/Symbol=20 l}_y}" font "Monospace, 25" offset 2

plot \
 "meta-hex_cuprum.gpd" u 1:3 w l ti "{/Monospace=20 численное решение}" lw 2 lt 1, \
 "jancovsky_cuprum.gpd" u 1:3 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
 "jancovsky_cuprum.gpd" u 1:5 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3


#set output "Теплопроводность вдоль оси Ox до 035 cuprum.png"
#set xrange [ 0.01 : 0.35 ]
#set ylabel "Коэффициент теплопроводности {{/Symbol=20 l}_x}" font "Monospace, 25" offset 2

#plot \
# "meta-hex_cuprum.gpd" u 1:2 w l ti "{/Monospace=20 численное решение}" lw 2 lt 1, \
# "jancovsky_cuprum.gpd" u 1:2 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
# "jancovsky_cuprum.gpd" u 1:4 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3#, \
 #"solution_janc_from_article.gpd" u 1:2 ti "" ps 4 lt 4


#set output "Теплопроводность вдоль оси Oy до 035 cuprum.png"
#set xrange [ 0.01 : 0.35 ]
#set ylabel "Коэффициент теплопроводности {{/Symbol=20 l}_y}" font "Monospace, 25" offset 2

#plot \
# "meta-hex_cuprum.gpd" u 1:3 w l ti "{/Monospace=20 численное решение}" lw 2 lt 1, \
# "jancovsky_cuprum.gpd" u 1:3 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
# "jancovsky_cuprum.gpd" u 1:5 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3


set output "Относительное отклонение решений по методу Янковского от численного вдоль оси Ox до 035 cuprum.png"
set xrange [ 0.01 : 0.35 ]
set ylabel "Относительное отклонение {{/Symbol=20 Dl}_{xx}}" font "Monospace, 25" offset 2

plot \
 "div_hex-janc_cuprum.gpd" u 1:2 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
 "div_hex-janc_cuprum.gpd" u 1:3 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3
 

set output "Относительное отклонение решений по методу Янковского от численного вдоль оси Oy до 035 cuprum.png"
set xrange [ 0.01 : 0.35 ]
set ylabel "Относительное отклонение {{/Symbol=20 Dl}_{yy}}" font "Monospace, 25" offset 2

plot \
 "div_hex-janc_cuprum.gpd" u 1:4 w l ti "{/Monospace=20 статический метод}" lw 2 lt 2, \
 "div_hex-janc_cuprum.gpd" u 1:5 w l ti "{/Monospace=20 кинематический метод}" lw 2 lt 3
  

set output "Среднее.png"
set xrange [ 0.01 : 1.0 ]
set ylabel "Относительное отклонение {{/Symbol=20 Dl}_{yy}}" font "Monospace, 25" offset 2

plot \
 "div_hex-janc_cuprum_midle.gpd" u 1:2 w l ti "{/Monospace=20 средний статический}" lw 2 lt 2, \
 "div_hex-janc_cuprum_midle.gpd" u 1:3 w l ti "{/Monospace=20 средний кинематический}" lw 2 lt 3, \
 "div_hex-janc_cuprum_midle.gpd" u 1:4 w l ti "{/Monospace=20 статический-кинематический}" lw 2 lt 4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
