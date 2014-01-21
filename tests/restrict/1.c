/*
 * =====================================================================================
 *
 *       Filename:  1.c
 *
 *    Description:  :
 *
 *        Version:  1.0
 *        Created:  13.08.2012 09:37:33
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

const uint64_t limit = 0xffff;

time_t test (uint8_t* restrict const des, const uint8_t* restrict const source)
{
    time_t timer = time(NULL);

   for(uint64_t j = 0; j < 0x1fffff; ++j) 
        for (uint64_t i = 0; i < limit; ++i)
        {
            des[i] = source[i] * source[i];
        };  

    return (time(NULL) - timer); 
};

int main(int argc, char *argv[])
{
    uint8_t* a = malloc(limit);
    uint8_t* b = malloc(limit);

    for (uint64_t i = 0; i < limit; ++i)
    {
        b[i] = i % 0xffff;
    };

    printf("%d %d %ld\n", a[24], b[15], test(a,b));
//    printf("%d %d %ld\n", a[24], b[15], test(a,b));
//    printf("%d %d %ld\n", a[24], b[15], test(a,b));

    free(a);
    free(b);

    uint8_t c[limit];
    uint8_t d[limit];

    for (uint64_t i = 0; i < limit; ++i)
    {
       d[i] = i % 0xffff;
    };

    printf("%d %d %ld\n", c[24], d[15], test(c,d));
//    printf("%d %d %ld\n", c[24], d[15], test(c,d));
//    printf("%d %d %ld\n", c[24], d[15], test(c,d));
    return 0;
}

