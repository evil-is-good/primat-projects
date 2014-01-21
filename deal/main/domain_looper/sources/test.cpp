/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  29.08.2012 10:57:53
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
#include <vector>

    template<int dim>
    struct Border
    {
        double coor[dim][2]; // 0 - black, 1 - white;
        double* operator[] (int i)
        {
            return coor[i];
        };
    };

int main(int argc, char *argv[])
{
    Border<2> b;
    b.coor[0][0] = 0;
    b.coor[0][1] = 1;
    b.coor[1][0] = 2;
    b.coor[1][1] = 3;
    printf("%f\n", b[0][0]);
    printf("%f\n", b[0][1]);
    printf("%f\n", b[1][0]);
    printf("%f\n", b[1][1]);

    std::vector<std::vector<int> > a;
    a .push_back (std::vector<int>(0));
    a[0] .push_back (10);
    a .push_back (std::vector<int>(0));
    a[1] .push_back (11);
    a[1] .push_back (12);
    printf("%d %d %d\n", a[0][0], a[1][0], a[1][1]);
    return 0;
}

