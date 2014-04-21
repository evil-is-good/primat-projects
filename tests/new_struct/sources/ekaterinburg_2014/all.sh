gnuplot "Ex.g"
gnuplot "Ez.g"
gnuplot "Nxy.g"
gnuplot "Nzx.g"
gnuplot "Nxz.g"
gnuplot "Mxy.g"
gnuplot "Mxz.g"
join mata-quadrate2.gpd mata-hexagon2.gpd | awk -f max_diff.awk > max_diff.gpd
