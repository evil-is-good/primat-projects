/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11.12.2012 13:16:30
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
//#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
//    unsigned char a[
//    4294967296];
//10485760];
//
#define FOR_I(begin, end) for(size_t i = begin; i < end; ++i)

//template <int dim>
//class Foo
//{
//    public:
//        Foo () 
//        {
//            for (int i = 0; i < 10; ++i)
//            {
//                x[i] = 1;
//                y[i] = 2;
//            };
//        };
//        int x[10];
//        int y[10];
//        template <int* a>
//        int bar (int index);
//
//        int aaa();
//};
//
//template <int dim>
//template <int* a>
//int Foo<dim>::bar (int index)
//{
//    return a[index];
//};
//
//template <int dim>
//int Foo<dim>::aaa ()
//{
//    return bar<x>(2);
//};

int foo(int a[3][3])
{
    a[0][0] = 11;
    return a[0][0];
};

class AAA
{
    public:
        AAA (int in) {i = in;};
        bool operator== (const AAA& b)
        {
            return (i == b.i);
        };
        operator int () {return i;};
    int i;
};

int main(int argc, char *argv[])
{

//    Foo<2> foo;
//    printf("%d %d\n", foo.x[2], foo.y[2]);
//    printf("%d\n", foo.aaa()); 
//    uint8_t* a = new uint8_t[4294967296];//3221225472];///2147483648];
//    for(size_t i = 0; i < 4294967296; ++i)
//        a[i] = i % 256; 
////    int i;
    
//    size_t esp_val = 0;

//    uint8_t a[1023*1024*8];//[5242880];//10485760];
//    uint8_t b[1024];//[5242880];
//    for(size_t i = 0; i < 4294967296; ++i)
//        a[i] = i % 256;
//    unsigned int esp_val = 0;
//    asm("movl %%esp, %0":"=d"(esp_val));
//    printf("%d\n", esp_val);
//    std::cin >> a[0];
//    std::cin >> a[11];
//    int a=10, b;
//    asm ("movl %1, %%eax; 
//            movl %%eax, %0;"
//            :"=r"(b)        /*  output */
//            :"r"(a)         /*  input */
//            :"%eax"         /*  clobbered register */
//        );    
//    asm("movl %ecx, %eax");

//    int a[3][3] = {10};
//
//    printf("%d\n", a[0][0]);
//    printf("%d\n", foo(a));
//    printf("%d\n", a[0][0]);
//
//    FOR_I(0, 10)
//    {
//        puts("123");
//    }

//    int i = 2;
//    int j = 2;
//
//    switch (i)
//    {
//        case j: puts("sdf"); break;
//    };

//    AAA aa(2);
//    AAA bb(2);

//    char* aa = "123";
//    const char bb[] = "123";
//
//    switch (aa)
//    {
//        case bb: puts("sdf"); break;
//    };

//    char* a = new char[10];//= {2};
//    std::vector<char> b;
//    b .push_back (10);
//    printf("%ld\n", sizeof(a));
//    delete [] a;


    std::ofstream of ("asd");
    of << 3.0 << std::endl;

    return 0;
}

