#include <stdlib.h>
#include <stdio.h>
#include "/home/primat/tmp/gomp/omp.h"
// #include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>

    // extern void init(float*, int);
    // extern void output(float*, int);
void simple(int n, float *a, float *b)

{

    int i;
    // init(a, n);
#pragma omp target map(a, b)
    {
#pragma omp parallel for 

    for (i=1; i<n; i++)
    {
        b[i] = (a[i] + a[i-1]) / 2.0;
        printf("%d\n", i);
    printf("%d %d %d\n", omp_get_num_devices(), omp_get_num_procs(), omp_get_num_threads());
    }
    }
    // output(b, n);

};


int main()
{
    float a[5] = {1,2,3,4,5};
    float b[5] = {5,4,3,2,1};
    printf("%d %d %d\n", omp_get_num_devices(), omp_get_num_procs(), omp_get_num_threads());

    simple(5, a, b);
    
    return EXIT_SUCCESS;
}
