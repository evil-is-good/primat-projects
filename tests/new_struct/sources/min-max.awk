BEGIN{min=0.0; max=0.0}; 
{if(NR>10){if($1>max){max=$3}; if($1<min){min=$3};};}; 
END{print min, max}
