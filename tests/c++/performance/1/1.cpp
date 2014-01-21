// #include <stdlib.h>
#include <stdio.h>
// #include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
// #include <dvec.h>
// #include <immintrin.h>

typedef double dbl;
typedef size_t st;
typedef const size_t cst;

cst max = 1000000;
// dbl A[max] __attribute__ ((aligned (16)));
// dbl B[max] __attribute__ ((aligned (16)));
// dbl C[max] __attribute__ ((aligned (16))); 
dbl A[max], B[max], C[max];

int main()
{
    for (st i = 0; i < max; ++i)
    {
        A[i] = 0;
        B[i] = i;
        C[i] = max - i;
    };
    // FOR (n, 0, 1000)
    for (st n = 0; n < 1000; ++n)
    {
        for (st i = 0; i < max; ++i)
        {
            A[i] = B[i] + C[i];
        };
    };
    // FOR (n, 0, 1000)
    // {
    //     for (st i = 0; i < max; i+=2)
    //     {
    //         __m128d b = _mm_load_pd(&B[i]);
    //         __m128d c = _mm_load_pd(&C[i]);
    //         __m128d a = _mm_add_pd(b, c);
    //         _mm_store_pd(&A[i], a);
    //     };
    // };

    printf("%f\n", A[10]);
    return 0;
}
