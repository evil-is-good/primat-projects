/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02.08.2012 14:49:44
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <iostream>

class A
{
    public:
        A() {n = 5; m = 15;};
        int n;
//        operator int () {return n;};
        operator int & () {return m;};
        typedef int am_;
        am_ am;
//        operator am_ & () {return am;};

    private:
        int m;
};

class B
{
    public:
        B () {n = 2;};
        const int length () {return n;};
        const int length (const int i) {n = i; return n;};
    private:
        int n;
};

class Report
{
    public:
        Report () {res = false;};
        bool res;
        operator bool & () {return res;};
};

//template<typename t>
//class the
//{
//    t the(t c) {return static_cast <t> (c);};
//};

#define the(t, c) (static_cast <t> (c))
#define of
#define of_the

#define is ==
#define then 

typedef bool& result;

int main(int argc, char *argv[])
{
    int n = 10;
    A a;
    n = static_cast<int&>(a);
    n++;
    static_cast<A::am_&>(a)++;
    the (int&, of a) ++;
    std::cout << n << " " << the (int&, of a) << std::endl;
    
    Report operation;
    the (result, of operation) = true;
    operation.res = true;

    if (the (result, of_the operation) is true) then
        std::cout << "Yes" << std::endl;

    B b;
    std::cout << b.length() << std::endl;
    b.length(5);
    std::cout << b.length() << std::endl;
    std::cout << ((3<4) and (5>3)) << std::endl;

    b.len();
    b.len(5);
    b .get_len();
    b .set_len(5);
    
//    the (power, of the (engine, of_the tractor_1));
//    (the (engine, of_the tractor_1)).power;
//    tractor_1.engine.power;

    return 0;
}

