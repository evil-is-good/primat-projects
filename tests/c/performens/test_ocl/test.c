#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define DY3(i, j, k) ( DIM_Y * (i) + (j) + DIM_X * DIM_Y * (k) )

#define DIM_X 256
#define DIM_Y 256
#define DIM_Z 256

#define SIZE 256*256*256 ///8388608

const double pi = 3.1415926535897932384626433832795;


/* struct timespec */
/* { */
/*     time_t   tv_sec;     */
/*     unsigned long long int tv_nsec;       */
/* }; */

struct timespec tm;

unsigned long long int getRealNanosecondsCount(struct timespec time);


double  u[SIZE], temp1[SIZE], temp2[SIZE],
               aim[SIZE], aip[SIZE], ajm[SIZE], ajp[SIZE], 
               akm[SIZE], akp[SIZE], con[SIZE];

int main(void)
{

    size_t i, j, k, l;

	for(k = 0; k < DIM_Z; ++k)
		for(i = 0; i < DIM_X; ++i)
	   		for(j = 0; j < DIM_Y; ++j)
			{
				u[DY3(i, j, k)] = 0.002 * pi + (j + i + k) % 2;
				con[DY3(i, j, k)] = 0.01 * (pi / 2.0) - (j + i + k) % 2; 

				aim[DY3(i, j, k)] = 0.11 * (pi / 3.0) - (j + i * 3 + k) % 2;
				aip[DY3(i, j, k)] = 0.13 * (pi / 3.1) + (2 * j + i + k) % 2;

				ajm[DY3(i, j, k)] = 0.21 * (pi / 1.05) - (j + i) % 2;
				ajp[DY3(i, j, k)] = 0.05 * (pi / 0.1) + (j + i) % 2;

				akm[DY3(i, j, k)] = 0.21 + (j + i + k) % 2; 
				akp[DY3(i, j, k)] = 0.7 * (pi / 1.05) - (3 * j + i + k) % 2;
		 	}


    unsigned long long int startTime1, endTime1, startTime2, endTime2;

    ////////////////////START TIMER//////////////////
    /* clock_gettime(0, &tm);  */
    /* startTime1 = getRealNanosecondsCount(tm); */
    time_t beg = time(NULL);
    /////////////////////////////////////////////////

    for (l = 0; l < 90; ++l)
	for(k = 0; k < DIM_Z; ++k)
		for(i = 0; i < DIM_X; ++i)
	   		for(j = 0; j < DIM_Y; ++j)
			{
                temp1[DY3(i, j, k)]  = con[DY3(i, j, k)]  +
									  0.5  * ajm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * ajp[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.5  * akm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * akp[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.5  * ajm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * ajp[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.5  * akm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * akp[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.5  * ajm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * ajp[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.5  * akm[DY3(i, j, k)] * u[DY3(i, j, k)] +
									  0.15 * akp[DY3(i, j, k)] * u[DY3(i, j, k)];
            }

    ///////////////////////FINISH TIMER///////////////
    printf("%d\n", time(NULL) - beg);
    /* clock_gettime(0, &tm);  */
    /* endTime1 = getRealNanosecondsCount(tm); */
    //////////////////////////////////////////////////


    /* printf("Elapsed time1: %llu nsec\n", endTime1 - startTime1); */


    ////////////////////START TIMER//////////////////
    /* clock_gettime(0, &tm);  */
    /* startTime2  = getRealNanosecondsCount(tm); */
    beg = time(NULL);
    /////////////////////////////////////////////////

    for (l = 0; l < 90; ++l)
   	for(i = 0; i < SIZE; ++i)
    {
        temp2[i]  = con[i]  + 0.5  * ajm[i] * u[i] +
                              0.15 * ajp[i] * u[i] +
                              0.5  * akm[i] * u[i] +
                              0.15 * akp[i] * u[i] +
                              0.5  * ajm[i] * u[i] +
                              0.15 * ajp[i] * u[i] +
                              0.5  * akm[i] * u[i] +
                              0.15 * akp[i] * u[i] +
                              0.5  * ajm[i] * u[i] +
                              0.15 * ajp[i] * u[i] +
                              0.5  * akm[i] * u[i] +
                              0.15 * akp[i] * u[i];
    }

    ///////////////////////FINISH TIMER///////////////
    printf("%d\n", time(NULL) - beg);
    /* clock_gettime(0, &tm);  */
    /* endTime2 = getRealNanosecondsCount(tm); */
    //////////////////////////////////////////////////

    /* printf("Elapsed time2: %llu nsec\n", endTime2 - startTime2); */


    double checkSum1 = 0.0;
	double checkSum2 = 0.0;

   	for(i = 0; i < SIZE; ++i)
    {
        checkSum1 += temp1[i];
		checkSum2 += temp2[i];
    } 

    printf("Check_sum1: %31.30E\nCheck_sum2: %31.30E\n", checkSum1, checkSum2);

    return 0;
}

unsigned long long int getRealNanosecondsCount(struct timespec time)
{
	return (unsigned long long int)time.tv_sec * 1000000000 + time.tv_nsec;
}
