/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.11.2012 11:35:33
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
#include <stdint.h>

#include "./1.h"

const uint8_t x = 0;
const uint8_t y = 1;
const uint8_t z = 2;

typedef double real;

double Nu[3][3] = 
//{{2,2,2},{2,2,2},{2,2,2}};
{{2,3,4},{5,6,7},{8,9,1}};
//{{0.25,0.25,0.25},{0.25,0.25,0.25},{0.25,0.25,0.25}};
//{{0.2,0.21,0.22},{0.23,0.24,0.25},{0.26,0.27,0.28}};

double a (uint8_t const i, uint8_t const j)
{
    return (1 - Nu[i][j] * Nu[j][i]);
};

double b (uint8_t const i, uint8_t const j, uint8_t const k)
{
    return (Nu[i][j] + Nu[i][k] * Nu[k][j]);
};

//double A (uint8_t const i)
//{
//    uint8_t k_1 = (k + 1) % 3;
//    uint8_t k_1 = (k + 2) % 3;
//
//    return ((a(k,k_1)*a(k,k_2) - b(y,z)*b(z,y)) / a(x,y));
//};

double const A = 
    1 - 
    (Nu[x][y]*Nu[y][x] + Nu[x][z]*Nu[z][x] + Nu[y][z]*Nu[z][y]) -
    (Nu[x][y]*Nu[y][z]*Nu[z][x] + Nu[x][z]*Nu[z][y]*Nu[y][x]);

int foo ()
{
    struct  
    {
        int operator() ()
        {
            return 2;
        };
    } aaa;

    return (aaa());
};

struct tensor
{
    double c[3][3];
};

tensor foo (tensor &a)
{
    tensor res;

    double A = 
        1 - 
        (a.c[x][y] * a.c[y][x] + a.c[z][x] * a.c[x][z] + a.c[y][z] * a.c[z][y]) -
        (a.c[x][y] * a.c[y][z] * a.c[z][x] + a.c[x][z] * a.c[z][y] * a.c[y][x]);

    for (uint8_t i = 0; i < 3; ++i)
    {
        int no_1 = (i + 1) % 3;
        int no_2 = (i + 2) % 3;

        for (uint8_t j = 0; j < 3; ++j)
        {
            if (i == j)
                res.c[i][j] = (1 - a.c[no_1][no_2] * a.c[no_2][no_1]);
            else
            {
                int k = (j == no_1) ? no_2 : no_1;
                res.c[i][j] = (a.c[j][i] + a.c[j][k] * a.c[k][i]);
            };

            res.c[i][j] *= (a.c[i][i] / A);
        };
    };
        
    return res;
};

tensor bar (tensor &a)
{
    tensor res;

    double A = 
        a.c[x][x] * a.c[y][y] * a.c[z][z] - 
        a.c[y][z] * a.c[z][y] * a.c[x][x] +
        a.c[x][y] * a.c[y][z] * a.c[z][x] - 
        a.c[y][x] * a.c[x][y] * a.c[z][z] - 
        a.c[y][y] * a.c[x][z] * a.c[z][x] +
        a.c[y][x] * a.c[x][z] * a.c[z][y]; 

    for (uint8_t i = 0; i < 3; ++i)
    {
        int no_1 = (i + 1) % 3;
        int no_2 = (i + 2) % 3;

        for (uint8_t j = 0; j < 3; ++j)
        {
            int k = (j == no_1) ? no_2 : no_1;

            if (i == j)
                res.c[i][j] = A;
            else
                res.c[i][j] = (a.c[i][j] * a.c[k][k] - a.c[i][k] * a.c[j][k]);

            res.c[i][j] /= 
                (a.c[no_1][no_1] * a.c[no_2][no_2] - 
                 a.c[no_1][no_2] * a.c[no_2][no_1]);
        };
    };
        
    return res;

};

int main(int argc, char *argv[])
{
    double E[3] = {1,5,3};//{1.0,2.0,3.0};
    double tau[3] = {6.2,2.4,7.0};
    double eps[3];

    eps[x] = (1 / E[x]) * tau[x] - 
        (Nu[y][x] / E[y]) * tau[y] - 
        (Nu[z][x] / E[z]) * tau[z]; 

    eps[y] = (1 / E[y]) * tau[y] - 
        (Nu[x][y] / E[x]) * tau[x] - 
        (Nu[z][y] / E[z]) * tau[z]; 

    eps[z] = (1 / E[z]) * tau[z] - 
        (Nu[y][z] / E[y]) * tau[y] - 
        (Nu[x][z] / E[x]) * tau[x]; 

    double new_tau[3] = {0.0, 0.0, 0.0};

    for (uint8_t N = 0; N < 3; ++N)
    {
        int no_1 = (N + 1) % 3;
        int no_2 = (N + 2) % 3;
//        printf("1=%d 2=%d\n", no_1, no_2);

        for (uint8_t i = 0; i < 3; ++i)
            if (i == N)
                new_tau[N] += a(no_1, no_2) * eps[i];
            else
                new_tau[N] += b(i, N, (i == no_1) ? no_2 : no_1) * eps[i];

        new_tau[N] *= (E[N] / A);
    };

    double a = 10.0;
    real b = 20.0;
    a = b;
    b = a;

//    new_tau = (a(z,y) * eps[x] + b(y,x,z) * eps[y] + 
//            b(z,x,y) * eps[z]) / A * E[x];

//    new_tau = (E[z] * (b(x,z,y)*eps[x] + b(y,z,x)*eps[y] + a(y,x)*eps[z])) / A;
    
//    double B = (a(x,y) * a(z,y) - (b(x,z,y) * b(z,x,y))) / E[x]; 
//
//    new_tau = (eps[x] * a(z,y) + (b(y,x,z)) * eps[y] +
//            b(z,x,y) * eps[z]) / A * E[x];

//    double B = 1 - Nu[y][z] * Nu[z][y] - Nu[x][z] * Nu[z][x] - 
//        Nu[z][x] * Nu[x][y] * Nu[y][z];
//
//    new_tau = b(x,z,y) * eps[x] + Nu[y][z] * eps[y] + eps[z];
//    new_tau *= E[z];
//    new_tau /= a(y,z) - Nu[z][x] * b(x,z,y);

//    double oppa = - (b(x,z,y) / E[x]) * tau[x] + a(y,z) / E[z] * tau[z] - Nu[y][z] * eps[y];
//    double oppa = (a(y,z) - Nu[z][x] * b(x,z,y)) / E[z] * tau[z] - 
//        Nu[y][z] * eps[y] - b(x,z,y) * eps[x];

//        ((Nu[x][z] + Nu[x][y] * (b(y,z)/a(x,y))) * E[z] / A) * eps[x] +
//        (((b(y,z)/a(x,y)) * E[z]) / A) * eps[y] +
//        (E[z] / A) * eps[z];

    printf("tau: %f %f %f\n", tau[x], tau[y], tau[z]);
    printf("eps: %f %f %f\n", eps[x], eps[y], eps[z]);
    printf("tau_new: %f %f %f\n", new_tau[x], new_tau[y], new_tau[z]);

    tensor fiz, nofiz, newfiz;

    fiz.c[x][x] = 1;
    fiz.c[x][y] = 0.25;
    fiz.c[x][z] = 0.25;
    fiz.c[y][x] = 0.5;
    fiz.c[y][y] = 2;
    fiz.c[y][z] = 0.5;
    fiz.c[z][x] = 0.75;
    fiz.c[z][y] = 0.75;
    fiz.c[z][z] = 3;

    nofiz = foo(fiz);

    newfiz = bar(nofiz);

    printf("\n");

    for (uint8_t i = 0; i < 3; ++i)
    {
        for (uint8_t j = 0; j < 3; ++j)
            printf("%f  ", fiz.c[i][j]);
        printf("\n");
    };

    printf("\n");

    for (uint8_t i = 0; i < 3; ++i)
    {
        for (uint8_t j = 0; j < 3; ++j)
            printf("%f  ", nofiz.c[i][j]);
        printf("\n");
    };

    printf("\n");

    for (uint8_t i = 0; i < 3; ++i)
    {
        for (uint8_t j = 0; j < 3; ++j)
            printf("%f  ", newfiz.c[i][j]);
        printf("\n");
    };
//    printf("A=%f B=%f\n", A, B * E[x]);
//    printf("oppa: %f\n", oppa);
//    printf("%d\n",foo());

//    double eps_1[3];
//
//    eps_1[x] = (1 / E[x]) * tau[x] - 
//        (Nu[y][x] / E[y]) * tau[y] - 
//        (Nu[z][x] / E[z]) * tau[z]; 
//
//    eps_1[y] = (1 / E[y]) * (1 - Nu[y][x] * Nu[x][y]) * tau[y] - 
//        (1 / E[z]) * (Nu[z][y] + Nu[z][x] * Nu[x][y]) * tau[z] -
//        Nu[x][y] * eps[x]; 
//
//    eps_1[z] = (1 / E[z]) * (1 - Nu[z][x] * Nu[x][z]) * tau[z] - 
//        (1 / E[y]) * (Nu[y][z] + Nu[y][x] * Nu[x][z]) * tau[y] -
//        Nu[x][z] * eps[x]; 
//
//    printf("eps_1: %f %f %f\n", eps_1[x], eps_1[y], eps_1[z]);
//
//    eps_1[y] = (1 / E[y]) * a(x,y) * tau[y] -
//        (1 / E[z]) * b(z,y) * tau[z] - 
//        Nu[x][y] * eps[x]; 
//
//    eps_1[z] = (1 / E[z]) * a(x,z) * tau[z] - 
//        (1 / E[y]) * b(y,z) * tau[y] -
//        Nu[x][z] * eps[x]; 
//
//    printf("eps_2: %f %f %f\n", eps_1[x], eps_1[y], eps_1[z]);

    return 0;
}

//class B
//{
//    public:
//        B(int k) {k_ = 1; printf("construct B\n");};
//        ~B() {printf("destroy B\n");};
//
//        int foo(int a) {return k_ + a + 10;};
//        int foo(int a) const {return k_*a*20;};
//
//        int k_;
//};
//
//void print_const(B* obj)
//{
//    const B* cObj = obj;
//    printf("%d\n", cObj->foo(10));
//}
//
//int main(int argc, char* argv[])
//{
//    B* cObj_const = new B(1);
//    B* cObj_const2= new B(2);
//
//    print_const(cObj_const);
//    print_const(cObj_const2);
//
//    delete cObj_const;
//    delete cObj_const2;
//
//    return 0;
//}

//struct B
//{
//    static int foo (int a) {a + 10;};
////    int foo (int a) const {a * 10;};
//};
//
//int main(int argc, char *argv[])
//{
//    printf("%d\n", B::foo(10));
//    return 0;
//}

