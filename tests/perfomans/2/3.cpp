#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int foo()
{
    return 10;
};

int main(int argc, char *argv[])
{
//    double a1 = atoi(argv[0])*1.1;
//    double a2 = atoi(argv[1])*1.1;
//    double a3 = atoi(argv[2])*1.1;

    int a1 = atoi(argv[1]);
    int a2 = atoi(argv[2]);
    int a3 = atoi(argv[3]);

//    printf("\n", `<args>`);
    time_t timer = time(NULL);
    unsigned long int i = 0;
    double a = 0;
    unsigned long int b = 0;
    for (; i < 0xFFFFFFFF; ++i)
    {
        b += a1;//(a1*a2+a3);
        b %= a2;
    }
//    double a = 3.0*4.5+7.8;
//    double a = fma(3.0,4.5,7.8);
    printf("%lf %ld %ld %d %d %d\n", a, b, (time(NULL)-timer), a1, a2, a3);
    return 0;
};

