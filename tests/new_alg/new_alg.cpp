/*
 * =====================================================================================
 *
 *       Filename:  new_alg.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10.01.2013 16:20:12
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

int main ()
{
    double x = 2.0;
    double y = 7.0;

    double a[2] = {1.0, 0.0};
    double b[2] = {0.0, 1.0};
    double c[2] = {a[0] * x + b[0] * y,
                   a[1] * x + b[1] * y};

    double v[3][2] = {{8.0, 2.0},
                      {7.0, 3.0},
                      {2.0, 4.0}};

    for (int i = 0; i < 2; ++i)
    {
        double p[3][2] = {{v[0][0], v[0][1]},
                          {v[0][0], v[0][1]},
                          {v[1][0], v[1][1]}};

        double k[3][2] = {{v[1][0], v[1][1]},
                          {v[2][0], v[2][1]},
                          {v[2][0], v[2][1]}};

        for (int j = 0; j < 3; ++j)
        {
           double t = 
               (c[i] - a[i] * k[j][0] - b[i] * k[j][1]) / 
               (a[i] * (p[j][0] - k[j][0]) + b[i] * (p[j][1] - k[j][1]));

           v[j][0] = k[j][0] + t * (p[j][0] - k[j][0]);

           v[j][1] = k[j][1] + t * (p[j][1] - k[j][1]);

//           printf("%f %f %f %f\n", t, v[j][0], p[j][0], 
//                   (a[i] * (p[j][0] - k[j][0]) + b[i] * (p[j][1] - k[j][1])));
        };
    };

    printf("%f %f %f\n", v[0][0], v[1][0], v[2][0]);
    printf("%f %f %f\n", v[0][1], v[1][1], v[2][1]);


    return 0;
}

