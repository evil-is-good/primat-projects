BEGIN{maxEx = 0.0; maxEz = 0.0; maxMxy = 0.0; maxMxz = 0.0; maxNxy = 0.0; maxNxz = 0.0; maxNzx = 0.0} \
{if ($2 > $13) {a = $2; b = $13} else {a = $13; b = $2}; if((a-b)/a>maxEx){maxEx=(a-b)/a}} \
{if ($10 > $21) {a = $10; b = $21} else {a = $21; b = $10}; if((a-b)/a>maxEz){maxEz=(a-b)/a}} \
{if ($11 > $22) {a = $11; b = $22} else {a = $22; b = $11}; if((a-b)/a>maxMxy){maxMxy=(a-b)/a}} \
{if ($12 > $23) {a = $12; b = $23} else {a = $23; b = $12}; if((a-b)/a>maxMxz){maxMxz=(a-b)/a}} \
{if ($3 > $14) {a = $3; b = $14} else {a = $14; b = $3}; if((a-b)/a>maxNxy){maxNxy=(a-b)/a}} \
{if ($8 > $19) {a = $8; b = $19} else {a = $19; b = $8}; if((a-b)/a>maxNxz){maxNxz=(a-b)/a}} \
{if ($4 > $15) {a = $4; b = $15} else {a = $15; b = $4}; if((a-b)/a>maxNzx){maxNzx=(a-b)/a}} \
#{if ($1 > $13) {print "1 ", $1} else {print "13 ", $13}}
#{print "2 ", $2; print "13 ", $13}
END{print maxEx, maxEz, maxMxy, maxMxz, maxNxy, maxNxz, maxNzx}
