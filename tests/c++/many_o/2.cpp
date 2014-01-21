/*
 * =====================================================================================
 *
 *       Filename:  2.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.07.2013 16:24:26
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
// #include "1.h"

extern int foo (int);
// template <typename T> class cfoo;
extern template class cfoo<int>;

int main()
{
    printf("%d\n", foo(2));
    cfoo<int> c;
    printf("%d\n", c());
    return 0;
};
