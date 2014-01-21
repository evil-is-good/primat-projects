dir_name = ['angle', 'piece', 'hole']
type_curve = ['move', 'stress_x', 'stress_y']
dir_name.each{|dir| type_curve.each{|t_c|
    (1..7).each{|n| 
        File.open("./#{dir}/#{n}_#{t_c}_curve_for_x.gpd", "w") do |f|
            f.puts `awk '{if($1 == 1.0) print $2, $3, $4}' ./#{dir}/#{n}_#{t_c}.gpd`.
            split("\n").uniq.sort_by{|s| s.split[0]}
        end
        File.open("./#{dir}/#{n}_#{t_c}_curve_for_y.gpd", "w") do |f|
            f.puts `awk '{if($2 == 1.0) print $1, $3, $4}' ./#{dir}/#{n}_#{t_c}.gpd`.
            split("\n").uniq.sort_by{|s| s.split[0]}
        end
    }}}
