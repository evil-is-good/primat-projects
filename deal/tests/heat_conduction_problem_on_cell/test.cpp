/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  20.09.2012 11:21:13
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

class A
{
    public:
        int n;
        virtual void foo () { n = 1;};
        virtual void bar () { n = 2;};
};

class B : public A
{
    public:
        int n;
        virtual void foo () { n = 3;};
};

int main(int argc, char *argv[])
{
    A a;
    B b;

    a .foo ();
    printf("%d\n", a.n);
    a .bar ();
    printf("%d\n", a.n);
    b .foo ();
    printf("%d\n", b.n);
    b .bar ();
    printf("%d\n", b.n);
    return 0;
}

