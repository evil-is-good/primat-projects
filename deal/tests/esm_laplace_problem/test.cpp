/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  21.09.2012 14:14:41
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
//#include <stdlib.h>
//#include <stdio.h>
//#include <stdint.h>
//#include <array>

//template <uint8_t dim>
//struct ContainerCoefficients
//{
//    typedef std::array<std::array<double, dim>, dim> T;
//};
//
//
//typedef std::array<double, 2> T;

//void operator, (int& a, int const& b)
//{
//    a = b;
//};

template <typename T>
inline T& _ASS (T& a, const T& b)
{
    return (a = b);
};

int main()
{
//    ContainerCoefficients<2>::T cc1;
//    ContainerCoefficients<2>::T cc2;
//
//    cc1.fill (2);
//    cc2.fill (3);
//    cc1.swap (cc2);
//    T cc;
//    cc.fill (2);

    int a = 1;
    int b = 2;

//    printf("%d\n", _ASS(a, b));
    _ASS(a, b);
//    a = b;

    return 0;
}

