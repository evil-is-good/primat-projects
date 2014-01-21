reset

WEIGHT = 1

set term png enhanced size 1280,1024#1280, 1024

set output "Эпюра перемешений вдоль оси Ox для отрезанного угла.png"
set origin 0.01, 0.01
set size 0.99, 0.99
#set xrange [ 0.01 : 0.99 ]
#set yrange [ 0.01 : 0.99 ]
#set xtics 0, 0.1, 1#
#set ytics 0.01 ,0.11 ,1
set key spacing 2.4
# \n Рис.3. Сравнение макротеплопроводности ячейки для \n включений различной формы
set xlabel "x" font "Monospace, 25"
set ylabel "Перемещения" font "Monospace, 25" offset 2

plot \
 "./angle/1_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 1}" lw 2, \
 "./angle/2_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 2}" lw 2, \
 "./angle/3_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 3}" lw 2, \
 "./angle/4_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 4}" lw 2, \
 "./angle/5_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 5}" lw 2, \
 "./angle/6_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 6}" lw 2, \
 "./angle/7_move_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 7}" lw 2
 

set output "Эпюра напряжений вдоль оси Ox для отрезанного угла.png"
set ylabel "Напряжения" font "Monospace, 25" offset 2

plot \
 "./angle/1_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 1}" lw 2, \
 "./angle/2_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 2}" lw 2, \
 "./angle/3_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 3}" lw 2, \
 "./angle/4_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 4}" lw 2, \
 "./angle/5_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 5}" lw 2, \
 "./angle/6_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 6}" lw 2, \
 "./angle/7_stress_x_curve_for_x.gpd" u 1:2 w l ti "{/Monospace=20 7}" lw 2
 


set xlabel "y" font "Monospace, 25"
set output "Эпюра перемешений вдоль оси Oy для бокового разреза.png"
set ylabel "Перемещения" font "Monospace, 25" offset 2

plot \
 "./piece/1_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./piece/2_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./piece/3_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./piece/4_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./piece/5_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./piece/6_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./piece/7_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "y" font "Monospace, 25"
set output "Эпюра напряжений вдоль оси Oy для бокового разреза.png"
set ylabel "Напряжения" font "Monospace, 25" offset 2

plot \
 "./piece/1_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./piece/2_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./piece/3_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./piece/4_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./piece/5_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./piece/6_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./piece/7_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "x" font "Monospace, 25"
set output "Эпюра перемешений вдоль оси Ox для бокового разреза.png"
set ylabel "Перемещения" font "Monospace, 25" offset 2

plot \
 "./piece/1_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 1}" lw 2, \
 "./piece/2_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 2}" lw 2, \
 "./piece/3_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 3}" lw 2, \
 "./piece/4_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 4}" lw 2, \
 "./piece/5_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 5}" lw 2, \
 "./piece/6_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 6}" lw 2, \
 "./piece/7_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "x" font "Monospace, 25"
set output "Эпюра напряжений вдоль оси Ox для бокового разреза.png"
set ylabel "Напряжения" font "Monospace, 25" offset 2

plot \
 "./piece/1_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./piece/2_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./piece/3_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./piece/4_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./piece/5_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./piece/6_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./piece/7_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2



  
set xlabel "y" font "Monospace, 25"
set output "Эпюра перемешений вдоль оси Oy для разреза по серёдке.png"
set ylabel "Перемещения" font "Monospace, 25" offset 2

plot \
 "./hole/1_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./hole/2_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./hole/3_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./hole/4_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./hole/5_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./hole/6_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./hole/7_move_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "y" font "Monospace, 25"
set output "Эпюра напряжений вдоль оси Oy для разреза по серёдке.png"
set ylabel "Напряжения" font "Monospace, 25" offset 2

plot \
 "./hole/1_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./hole/2_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./hole/3_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./hole/4_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./hole/5_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./hole/6_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./hole/7_stress_y_curve_for_x.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "x" font "Monospace, 25"
set output "Эпюра перемешений вдоль оси Ox для разреза по серёдке.png"
set ylabel "Перемещения" font "Monospace, 25" offset 2

plot \
 "./hole/1_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 1}" lw 2, \
 "./hole/2_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 2}" lw 2, \
 "./hole/3_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 3}" lw 2, \
 "./hole/4_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 4}" lw 2, \
 "./hole/5_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 5}" lw 2, \
 "./hole/6_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 6}" lw 2, \
 "./hole/7_move_curve_for_y.gpd" u 1:3 w p ti "{/Monospace=20 7}" lw 2
  
  
set xlabel "x" font "Monospace, 25"
set output "Эпюра напряжений вдоль оси Ox для разреза по серёдке.png"
set ylabel "Напряжения" font "Monospace, 25" offset 2

plot \
 "./hole/1_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 1}" lw 2, \
 "./hole/2_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 2}" lw 2, \
 "./hole/3_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 3}" lw 2, \
 "./hole/4_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 4}" lw 2, \
 "./hole/5_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 5}" lw 2, \
 "./hole/6_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 6}" lw 2, \
 "./hole/7_stress_y_curve_for_y.gpd" u 1:3 w l ti "{/Monospace=20 7}" lw 2
  
