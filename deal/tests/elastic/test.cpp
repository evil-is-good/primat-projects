/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  14.09.2012 09:56:26
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

template <typename T>
inline void foo (T& a, const T& b)
{
    a = b;
//    return ({a = b; a;});
};

//template int foo (int, int);

class A
{
    public:
        A () {};
        virtual int foo () {return 0;};
        virtual int bar () {return 0;};
};

class B : public A
{
    public:
        B () {};
        virtual int foo () {return 1;};
};

class C : public B
{
    public:
        C () {};
        virtual int foo () {return 2;};
};

int main(int argc, char *argv[])
{
    A a;
    B b;
    C c;

    printf("%d %d %d\n", a.foo(), b.foo(), c.foo());
    printf("%d %d %d\n", a.bar(), b.bar(), c.bar());
    int n = 10;
    float m = 20;
    foo(n,m);
    printf("%d\n", n);

    return 0;
}

