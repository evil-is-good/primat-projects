#include <stdlib.h>
#include <stdio.h>
#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>

#define SUM(I, BEGIN, END, BODY) ({dbl tmp = 0.0; for (st I = BEGIN; I < END; ++I) {tmp += BODY;}; tmp;});

int main()
{
    int a = ({int tmp = 10; decltype(tmp) a; 12;});
    printf("%d\n", a);
    dbl S1 = SUM(i, 1, 4, i);
    printf("%lf\n", S1);
    puts("sdfvdfv");
    
    return EXIT_SUCCESS;
}
