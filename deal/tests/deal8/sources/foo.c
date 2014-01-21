/*
 * =====================================================================================
 *
 *       Filename:  foo.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  22.08.2012 11:35:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
//#include <stdlib.h>
//#include <time.h>
//#include <stdio.h>
//
//
//int foo()
//{
//  srand (time (NULL));
//  int res = rand() % 10;
//  return res;
//};
//
//int main(int argc, char *argv[])
//{
//    const int dim = 2;
//        struct 
//        {
//            size_t index[dim];
//            double coor[dim];
//        } pre_black[3], pre_white[3];
//    int mmm[foo()];
//    __int128_t a;
//    a = 0x10000000000000000;//ffffffffffffffff;
//    return 0;
//}

struct AOA
{
    char a;
    char c;
    int b;
} __attribute__((aligned(2)));

#include <stdio.h>
#include <xmmintrin.h>

__m128 add128(__m128 a, __m128 b)
{
     __m128 r = _mm_add_ps(a, b);
      return r;
}

float  A[4] __attribute__((aligned(16))) = { 2.0f, -1.0f, 3.0f, 4.0f};
float  B[4] __attribute__((aligned(16))) = {-1.0f,  3.0f, 4.0f, 2.0f};
float  C[4] __attribute__((aligned(16))) = { 0.0f,  0.0f, 0.0f, 0.0f};

int main()
{
     __m128 a = _mm_load_ps(&A[0]);
     __m128 b = _mm_load_ps(&B[0]);

     __m128 c = add128(a,b);
     _mm_store_ps(&C[0], c);

     printf("%f %f %f %f\n", C[0], C[1], C[2], C[3]);

     struct AOA aoa;

     printf("%ld %ld\n", sizeof(aoa), sizeof(int));

     struct 
     {
         char a;
         int  b;
     }* AFM;

     
     struct 
     {
         int a;
         int  b;
     }* BFM;

     AFM = new typeof(AFM) [10]; 


     AFM[5].a = 10;

     delete AFM;

     return 0;

}
