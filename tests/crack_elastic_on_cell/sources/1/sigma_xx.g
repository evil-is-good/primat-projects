reset

pow(x,d1,d2) = x < 0.5 ? x < 0.25 ? x**d1 : x**d2 : x > 0.75 ? x**d1 : x**d2

f1(x,d1,d2) =  x > 0.5 ? 1.0 : pow((x) * 2.0, d1, d2)
f2(x,d1,d2) =  x > 0.5 ? pow((1.0 - x) * 2.0, d1, d2) : pow((x) * 2.0, d1, d2)
f3(x,d1,d2) =  x < 0.5 ? 1.0 : pow(2.0 * (1 - x), d1, d2)

set xrange [0:1.0]
set yrange [0:1.0]
set xtics 0,0.1,1.0
set ytics 0,0.1,1.0
set size 1.0, 1.0
set term png enhanced size 1024, 1024
unset key
set pm3d map
set xlabel "X"
set ylabel "Y"

#gray > 0.25 ? gray < 0.75 ? (gray-0.25)*2.0 : 1.0 : 0.0  (((0.5-gray)*8.0)**0.5)/4.0
#set palette rgbformulae 30, 31, 32
#set palette function gray > 0.5 ? gray-0.5 : 0.0, gray > 0.5 ? gray-0.5 : 0.0, gray**0.8
#set palette function 0.0, gray/1.5, 1.0 - (gray/4.0) gray > 0.5 ? (gray - 0.5)/1.0 : 0.0
set palette function gray > 0.5 ? (((gray-0.5)*8.0)**0.5)/4.0 : 0.0, gray > 0.5 ? (gray-0.5)*2.0 : 0.0 , gray > 0.5 ? gray*2.0 : gray*2.0
#set palette function gray, gray, gray
#set palette function gray > 0.5 ? (1.0 - gray)*0.5 : gray*0.5, gray > 0.5 ? (gray-0.5)*2.0 : 0.0 , gray > 0.5 ? gray*2.0 : gray*2.0

#set cbrange [-1.3:1.3]

set title "Stress {/Symbol s}_{xx} for i = 2" font "Serif, 30"
set output "sigma_xx.png"
splot "sigma_x.gpd" using ($2):($1):($3) with pm3d

#set title "Stress {/Symbol s}_{2} for R = 0.3" font "Serif, 30"
#set output "main_stress_2.png"
#splot "main_stress.gpd" using ($2):($1):($4) with pm3d

set palette function 1.0-gray, 1.0-gray, 1.0

set title "Stress {/Symbol s}_{xy} for R = 0.3" font "Serif, 30"
set output "sigma_xy.png"
splot "sigma_y.gpd" using ($2):($1):(abs($3)) with pm3d

D1 = 2.0
D2 = 0.5#f1(gray, D1, D2), f2(gray, D1, D2), f3(gray, D1, D2)

set palette function \
 gray > 0.5 ? 1.0 : (gray) * 2.0, \
 gray > 0.5 ? (1.0 - gray) * 2.0: (gray) * 2.0, \
 gray < 0.5 ? 1.0 : 2.0 * (1 - gray)
 
#set palette function gray > 0.5 ? ((gray-0.5)*2.0)**0.5 : 0, 0, gray < 0.5 ? (2.0*(0.5-gray))**0.5 : 0


set title "Stress {/Symbol s}_{yy} for i = 25" font "Serif, 30"
set output "sigma_yy.png"#"../res_yy.gpd" u ($2):($1):($4) w pm3d \
set zrange [-0.15:0.15] # "../sigma_y.gpd" "../main_stress0.gpd"
splot \
"../brocken_cells_in_line.gpd" u ($2):($1):($3) w p pt 5 \
# "../brocken_cells_0.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_1.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_2.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_3.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_4.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_5.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_6.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_7.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_8.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_9.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_10.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_11.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_12.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_13.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_14.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_15.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_16.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_17.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_18.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_19.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_20.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_21.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_22.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_23.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_24.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_25.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_26.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_27.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_28.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_29.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_30.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_31.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_32.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_33.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_34.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_35.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_36.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_37.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_38.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_39.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_40.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_41.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_42.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_43.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_44.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_45.gpd" u ($2):($1):($3) w p pt 5 lc rgb "#00dd00" title "16" \
#,"../brocken_cells_46.gpd" u ($2):($1):($3) w p pt 5 lc rgb "#770066" title "7" \
#,"../brocken_cells_47.gpd" u ($2):($1):($3) w p pt 5 lc rgb "#660077" title "8" \
#,"../brocken_cells_48.gpd" u ($2):($1):($3) w p pt 5 lc rgb "#550088" title "9" \
#,"../brocken_cells_49.gpd" u ($2):($1):($3) w p pt 5 lc rgb "#440099" title "10" \
# ,"../brocken_cells_00.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_01.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_02.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_03.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_04.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_05.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_06.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_07.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_08.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_09.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_010.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_011.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_012.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_013.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_014.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_015.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_016.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_017.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_018.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_019.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_020.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_021.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_022.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_023.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_024.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_025.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_026.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_027.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_028.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_029.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_030.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_031.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_032.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_033.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_034.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_035.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_036.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_037.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_038.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_039.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_040.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_041.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_042.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_043.gpd" u ($2):($1):($3) w p pt 5 \
#  ,"../brocken_cells_044.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_045.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_046.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_047.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_048.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_049.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_050.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_051.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_052.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_053.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_054.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_055.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_056.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_057.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_058.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_059.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_060.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_061.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_062.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_063.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_064.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_065.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_066.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_067.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_068.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_069.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_070.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_071.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_072.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_073.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_074.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_075.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_076.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_077.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_078.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_079.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_080.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_081.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_082.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_083.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_084.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_085.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_086.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_087.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_088.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_089.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_090.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_091.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_092.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_093.gpd" u ($2):($1):($3) w p pt 5 \
# ,"../brocken_cells_094.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_075.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_076.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_077.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_078.gpd" u ($2):($1):($3) w p pt 5 \
#,"../brocken_cells_079.gpd" u ($2):($1):($3) w p pt 5 \
#


#set title "Stress {/Symbol s}_{1} for R = 0.3" font "Serif, 30"
#set output "main_stress_1.png"
#splot "main_stress.gpd" using ($2):($1):($3) with pm3d
#unset pm3d

