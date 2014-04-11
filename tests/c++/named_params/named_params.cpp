#include <stdlib.h>
#include <stdio.h>
#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>


struct foo
{
    const int a;
    const int b;
    int operator() () {return a + b;};
};

// struct PS
// {
//     int a = 0;
//     int b = 0;
// };
// 
// int foo2(const PS ps)
// {
//     return ps.a + ps.b;
// };

int main()
{
    foo{a : 10, 15};
    printf("%d\n", foo{.a = 10, b : 15}());
    return EXIT_SUCCESS;
}
