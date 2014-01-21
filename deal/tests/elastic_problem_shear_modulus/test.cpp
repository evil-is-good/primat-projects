/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09.10.2012 10:11:12
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

namespace A
{
    int foo () {return 0;};
};

typedef int int3; 

template <int dim>
class B
{
    public:
    int foo () {return dim;};
    typedef int int2[dim]; 
};

// gim.elmi
// wralhefrek

int main(int argc, char *argv[])
{
//    int3 a;
    B<2>::int2 a;
    a[0] = 10;
//    printf("%d %d\n", A::foo(), B<2>::foo());
    return 0;
}

