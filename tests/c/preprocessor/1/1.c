#include "stdio.h"

int foo(int n)
{
    #if (n < 2)
        return 0;
    #else
        return 1;
    #endif    
}

int main()
{
    #ifdef __STDC_LIMIT_MACROS
        printf("yes\n");
    #else
        printf("no\n");
    #endif
    printf("%d\n",foo(10));
    return(foo(10));
}
