#include <stdlib.h>
#include <stdio.h>
#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>

template <template<typename, std::size_t> class T, st n>
class A
{
    public:
        int i;
        T<u8, n> e; 
};


int main()
{
    A<arr, 2> a;
    a.e[1] = 10;
    printf("%d\n", a.e[1]);
    printf("%.100f %.100f\n", 1.0, 2.0);
    dbl b[2] = {1.1, 2.0};
    // b[0] = 1.0;
    printf("%.100f %.100f\n", b[0], b[1]);
    
    return EXIT_SUCCESS;
}
