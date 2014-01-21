// int bar ()
// {
//     return 10;
// };
// 
// int main ()
// {
//     int foo ()
//     {
//         return 1;
//     };
// 
//     //foo = &bar;
//     foo();
//     return 0;
// };

#include <stdio.h>

typedef int b_t;
typedef int c_t;
typedef int boundary_id_t;
typedef int material_id_t;

struct foo
{
    int a[2];
    int e;
    union  
    {
        b_t b;
        c_t c;
    } ;
};

union bar
{
    int a;
    int b;
};

template <int structdim>
struct CellData
{
    unsigned int vertices[2];

    union
    {
        boundary_id_t boundary_id;
        material_id_t material_id;
    };
};
   

int main ()
{
    // foo f;
    // f.u.b = 10;
    foo f{{1, 2}, 10,  {.b = 3}};
    CellData<1> g {1, 2, {.boundary_id = 3}};
    // foo f;
    // f.a[0] = 0;
    // f.a[1] = 1;
    // f.e = 2;
    // f.u.b = 3;
    // puts("ssdf");
    printf("%d %d %d %d\n", f.a[0], f.a[1], f.e, f.c); 
    printf("%d\n", g.boundary_id); 
    return 0;
}
