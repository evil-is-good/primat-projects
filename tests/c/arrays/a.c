/*
 * =====================================================================================
 *
 *       Filename:  a.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09.02.2013 14:14:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdio.h>
int main ()
{
    unsigned char a[2][2] = {{1, 2}, {3, 4}};
    printf("%d, %d\n", *(*(a + 1) + 1), a[1][1]);
}
