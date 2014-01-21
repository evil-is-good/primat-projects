#include <stdlib.h>
#include <stdio.h>
#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
// #include <dvec.h>
#include <immintrin.h>



int main()
{
    st max = 100;
    double A[max], B[max], C[max]; 
    for (st i = 0; i < max; ++i)
    {
        A[i] = 0;
        B[i] = i;
        C[i] = max - i;
    };
    for (st i = 0; i < max; ++i)
    {
        A[i] = B[i] + C[i];
    };

    
    return EXIT_SUCCESS;
}
