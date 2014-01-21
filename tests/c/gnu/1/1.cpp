/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  27.07.2012 10:05:17
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

//int foo1(int a)
//{
//    int foo2(int b)
//    {
//        return a + b;
//    }
//
//    return foo2(10);
//}

class Foo {
    int a;
    char b;
    int c;
    char d;
};

class Bar {
    int a;
    int c;
    char b;
    char d;
};

#include <iostream>
#include <math.h>

int main(int argc, char *argv[])
{
//    printf("%d\n", foo1(15));
    std::cout << sizeof(Foo) << " " << sizeof(Bar) << std::endl;
    if ((2 < 3) and (4 > 2))
        std::cout << "YES " << sizeof(size_t) << fma(2,3,1) << std::endl;
    return 0;
}

