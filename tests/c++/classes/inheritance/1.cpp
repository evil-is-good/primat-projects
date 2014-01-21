/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15.10.2013 14:50:42
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
#include <array>
#include "/home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h"
#include <functional>

class A
{
    public:
        void f1 () {};
    protected:
        void f2 () {};
    private:
        void f3 () {};
};

class B : public A
{
    public:
        // void foo () {f1(); f2(); f3();};

};

// struct O;
// struct P : O{};

struct O
{
    public:
        char i;
        char j;
        void f1(){m[0] = 10;};
        O& p() {return *this;};
        friend class P;
    private:
        char a;
        char m[10];
    protected:
        char b;
};

struct P : O
{
    public:
        void f() {i = 1; j = 2; b = 4; a = 3;};
        short i;
        void f2(){f1();};
        friend O;

};

class C
{
    public:
        C () : a(20) {};
        C (int i) : a(i) {};
        virtual void foo () {puts("C foo");};
        int a;
};

class D : public C
{
    public:
        // C c(i);
        D () {};//{::C();};
        // D(int i) : C(i) { puts ("construct D");}; 
        D(int i) {::C(10); puts ("construct D");}; 
        virtual void foo () {puts("D foo");};
};

// class St
// {
//     public:
//         int static f1(){a = 2; b = 3; return 10;};
//         int f2(){return 10;};
//         static int a;
//         int b;
// 
// };
// 
// int St::a;


class B1
{
    protected:
        int m;
    public:
        friend class A1;
};

class B2 : public B1
{
    public:
        int& foo () {return m;};
};

class A1
{
    public:
        B1 b1;
    public:
        int& foo () {return b1.m;};
};

template <typename T>
void assemble_matrix (int j, T i)
{

};

class A3
{
    public:
        int a;
        A3 (int a) {this->a = a;};
        ~A3 () {std::cout << "destr " << a << std::endl;};
};

template <int olo>
class A4
{
    public:
        virtual int foo () = 0;
        virtual int bar () = 0;
};

class B4 : public A4<2>
{
    public:
        virtual int foo () override { return d; };
        virtual int bar () { return 10; };
        int d = 12;
};
template <int olo>
class C4 : public A4<olo>
{
    public:
        virtual int foo () { return e; };
        virtual int bar () { return 10; };
        int e = 15;
};

int foo4 (A4<2> &a)
{
    return a.foo();
};

namespace NS1
{
    void a(){ int e;};
    namespace NS2
    {
    void b()
    {
        a();
        // ::a();
        ::NS1::a();
    };
    };
    void c()
    {
        // b();
        NS2::b();
        NS1::NS2::b();
        // ::NS2::b();
        ::NS1::NS2::b();
    };
};

std::function<size_t(size_t)> operator "" _plus(unsigned long long int a)
{
    return [a](size_t b){return a+b;};
};

class A5
{
    class B5 {};
    static void foo6 () {};
    class C5 {B5 b5; void bar6() {A5::foo6;};};
};

template <st a, st b>
class A6
{

};

template <>
class A6<1, 2>
{
    public:
    A6(){};
    void foo ();
};

// template<st b>
// A6<1, b>::A6(){};

// template<>
int A6<1, 2>::foo(){};

// template<>
// class A8<2>{};

int main ()
{
    A5 a5;
    A6<1,2> a6;
    a6.foo();
    // printf("%ld\n", 10_plus(20_plus(30)));
    // int a = 10;
    // printf("%d\n", a);
    // printf("%d\n", a++);
    // printf("%d\n", ++a);
    // printf("%d\n", a);
    // B4 b4; 
    // C4<2> c4;
    // std::cout << foo4(c4) << std::endl;
    // B2 b2;
    // b2.foo() = 10;
    // B1 b1;
    // b1 = b2;
    // A1 a1;
    // a1.foo() = 20;
    // assemble_matrix(1, A1());

    // A3 a3(10);
    // std::cout << a3.a << std::endl;
    // puts("1");
    // A3(11);
    // std::cout << A3(12).a << std::endl;
    // puts("2");
    // A a;
    // B b;
    // a.f1();
    // a.f2();
    // a.f3();
    // b.f1();
    // b.f2();
    // b.f3();

    // O o;
    // o.i = 10;
    // o.j = 20;
    // P p;
    // p.i = 30.0;
    // p.j = 0;
    // p.p().j = 10;

    // // printf("%ld %d %d\n",sizeof(p), *((char*)(&o))+0, p.i);
    // printf("%ld %ld\n", sizeof(o.i), sizeof(p.i));
    // printf("%ld %ld\n", sizeof(o), sizeof(p));
    // printf("%d\n", p.j);
    // printf("%ld %ld\n", sizeof(p.i), sizeof(p.p().i));
    // printf("%d %d\n", p.i, p.p().i);

    // // typeof(o.i);

    // C c;
    // D d(10);
    // c.foo();
    // d.foo();
    // printf("%d %d\n", c.a, d.a);

    // // printf("%d\n", St::f1());
    // printf("%ld %ld %ld\n", sizeof(float), sizeof(double), sizeof(long double));

    return 0;
}
