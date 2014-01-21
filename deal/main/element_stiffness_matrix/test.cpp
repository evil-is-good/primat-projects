/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07.09.2012 16:11:21
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include "./test.h"

template< int dim>
class B : public A<dim,int>
{
    public:
        virtual int a ();// { return 0; };
};

template< int dim>
int B<dim>::a ()
{
    return 0;
};

int main(int argc, char *argv[])
{
    B<10> a;
    a.a();
    a.b();

    return 0;
}

