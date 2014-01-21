/*
 * =====================================================================================
 *
 *       Filename:  2.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  27.07.2012 14:28:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* void foo1(uint64_t steps, uint64_t sec)
{
    time_t timer = time(NULL);

    double a = 0;
    uint64_t i = 0;
    for (; i <= steps; ++i)
    {
        a = 3.1 * 4.2 + (i*0.5);
        if ((time(NULL) - timer) >= sec)
            break;
    }
    
    printf("time=%ld steps=%ld a=%lf\n", (time(NULL)-timer), i, a);

}

void foo2(uint64_t steps, uint64_t sec)
{
    time_t timer = time(NULL);

    double a = 0;
    uint64_t i = 0;
    for (; i <= steps; ++i)
    {
        a = fma(3.1, 4.2, (i*0.5));
        if ((time(NULL) - timer) >= sec)
            break;
    }
    
    printf("time=%ld steps=%ld a=%lf\n", (time(NULL)-timer), i, a);

}*/

//void foo(int n)
//{
//    int i[n];
//}

#include <stdbool.h>
#include <stdlib.h>
//#include "brainfuck.lol"

struct foo
{
    int a;
    int b;
};

int main(int argc, char *argv[])
{
//    uint64_t sec = 55554;
//    uint64_t steps = 0x2FFFFFF;
    //bool a;

//    int a;
//    int b = 4;
//    int c = 5;
//    int g = 10;
//    int dr[3] = {1,2,3};
    int i = 1;
//    int j = 2;
    //a = ((c!=b)??!??!(a<b))?(<%dr??(i:>++;dr<:i??);%>):(b<g?(??<a--;c;??>):dr??(j??));

   i++,i++,i++;

--i; 
#define with_params
#define and_assigned_to
report = GridGenerator::create_grid(tria, with_params 1,2,0);
report = GridGenerator::create_grid(1,2,0, and_assigned_to tria);
struct foo F;
i = F .a;

    
//    _Bool b = true;
/* 
    foo2(steps,sec);
    foo1(steps,sec);
   */ 
    return 0;
}

