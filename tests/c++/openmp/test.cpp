/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  21.12.2012 08:56:44
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

    double a[1024*1024] = {0};
int main ()
{
    time_t start = time(NULL); 
size_t i = 0;
size_t j = 0;
size_t k = 0;
//#pragma omp parallel for default(shared) private(i, k)
//#pragma omp parallel for
    for (i = 0; i < 1024*5 ; ++i)
    for (j = 0; j < 1024*1024; ++j)
    for (k = 0; k < 2; ++k)
        a[i] = (j % 1024) * 5.0 + k;

    printf("%f %ld\n", a[10], (time(NULL) - start));
    return 0;
}
