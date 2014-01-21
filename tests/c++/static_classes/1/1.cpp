/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  :q
 *
 *        Version:  1.0
 *        Created:  26.07.2012 15:58:31
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
#include <iostream>

class AA
{
    public:

    AA(){a = 10;}
    void operator()(void (*a)()){a();}
    int a;
    void set_b(int n){b = n;}
    private:
    int b;
};

int foo()
{
    return 11;
}

void foo2()
{
    printf("YES\n");
}

enum {Aa,Ab,Ac};
enum etra {Oa,Ob,Oc};

int main(int argc, char *argv[])
{
//    int (*dfg)() = foo;
    AA aa;
    aa(foo2);
//    void ( AA::*func )(int) = &AA.set_b;

    enum etra ett = Oc;

    std::cout << Ab << "\n" << Ob << "\n" << ett << std::endl;
    std::cout << ({int a = 7; a;}) << std::endl;
    int a = (
            {
                int res = 1;
                for(int i = 0; i < 10; i++)
                    res *= 2;
                res;
            }
            );
    std::cout << a << std::endl;

//    abc dfg = foo;
//    printf("%d\n", dfg());
    return 0;
}

