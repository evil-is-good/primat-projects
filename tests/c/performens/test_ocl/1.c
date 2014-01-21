//#include <stdio.h>
//#include <stdlib.h>


#define DY3(i, j, k) ( DIM_Y * (i) + (j) + DIM_X * DIM_Y * (k) )

#define DIM_X 128
#define DIM_Y 256
#define DIM_Z 128

#define SIZE 8388608

//const double pi = 3.1415926535897932384626433832795;


int main(void)
{

	static double  u[SIZE], temp1[SIZE], temp2[SIZE],
    	    	   aim[SIZE], aip[SIZE], ajm[SIZE], ajp[SIZE], 
    		       akm[SIZE], akp[SIZE], con[SIZE];

/*
	for(int k = 0; k < DIM_Z; ++k)
		for(int i = 0; i < DIM_X; ++i)
	   		for(int j = 0; j < DIM_Y; ++j)
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


*/

   	for(int i = 0; i < SIZE; ++i)
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

    return 0;
}
