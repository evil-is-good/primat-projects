/*
 * =====================================================================================
 *
 *       Filename:  str-test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  19.10.2012 11:30:43
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

struct A
{
    int a;
    int b;
};

int main(int argc, char *argv[])
{
    A a,b;
    a.a = 1;
    a.b = 2;
    b = a;
    a.a = 7;
    b.b = 10;
    printf("%d %d %d %d\n", a.a,a.b,b.a,b.b);
    return 0;
}

