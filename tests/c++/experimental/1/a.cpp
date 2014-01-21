/*
 * =====================================================================================
 *
 *       Filename:  a.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  30.01.2013 12:45:43
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdio.h>
#include <stdint.h>

template <typename T> 
class Int
{
    T i;
    
    public:

    Int() : i(0) {};

    explicit operator T () {return i;};
    operator double () = delete;
    operator float () = delete;
};

template <typename T> 
class Float
{
    T i;
    
    public:

    Float() : i(0.0) {};

    operator T () {return i;};
    operator int8_t () = delete;
    operator int16_t () = delete;
    operator int32_t () = delete;
    operator size_t () = delete;
    operator uint8_t () = delete;
    operator uint16_t () = delete;
    operator uint32_t () = delete;
};

void foo (size_t i)
{

};

template<typename T1, typename T2>
T1 not_cast (const T2& a)
{ // Только C++11
    return T1({a});
};

template <typename T> 
class StrongType
{
    T i;
    
    public:

    StrongType ()    : i(0) {};
    StrongType (T o) : i(o) {};

    explicit operator T () {return i;};

//    StrongType<T>& operator = (T o) {i = o; return *this;};
};

#define DEVELOP

#ifdef DEVELOP

//#define i32  Int<int>
//#define imax Int<size_t>
//
//#define f1 Float<float>
//#define f2 Float<double>

#define i8  StrongType<int8_t> 
#define i16 StrongType<int16_t>
#define i32 StrongType<int32_t>
#define i64 StrongType<int64_t>
#define u8  StrongType<uint8_t> 
#define u16 StrongType<uint16_t>
#define u32 StrongType<uint32_t>
#define u64 StrongType<uint64_t>

#define ci8  const StrongType<int8_t> 
#define ci16 const StrongType<int16_t>
#define ci32 const StrongType<int32_t>
#define ci64 const StrongType<int64_t>
#define cu8  const StrongType<uint8_t> 
#define cu16 const StrongType<uint16_t>
#define cu32 const StrongType<uint32_t>
#define cu64 const StrongType<uint64_t>

#define sst StrongType<size_t>
#define csst const StrongType<size_t>

#define sfloat   StrongType<float>
#define sdouble  StrongType<double>
#define sldouble StrongType<long double> 

#define csfloat   const StrongType<float>
#define csdouble  const StrongType<double>
#define csldouble const StrongType<long double> 

#else

#define i8  int8_t 
#define i16 int16_t
#define i32 int32_t
#define i64 int64_t
#define u8  uint8_t 
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#define ci8  const int8_t 
#define ci16 const int16_t
#define ci32 const int32_t
#define ci64 const int64_t
#define cu8  const uint8_t 
#define cu16 const uint16_t
#define cu32 const uint32_t
#define cu64 const uint64_t

#define sst size_t
#define csst const size_t

#define sfloat   float
#define sdouble  double
#define sldouble long double 

#define csfloat   const float
#define csdouble  const double
#define csldouble const long double 

#endif

int main()
{
    i32 a;

    foo(10);
    foo(10.0);

//    foo(a);

//    StrongType<int8_t> a1;
//    StrongType<int16_t> a2;
//    StrongType<int32_t> a3;
//    StrongType<float> a1;
//    StrongType<double> a2;
//    StrongType<long double> a3;

//    sfloat a1;
//    sdouble a2;
//    sldouble a3;
//    
//
//    a1 = a1;
//    a1 = a2;
//    a1 = a3;
//    a2 = a1;
//    a2 = a2;
//    a2 = a3;
//    a3 = a1;
//    a3 = a2;
//    a3 = a3;

    sst a1 = 10;
    size_t b = size_t(a1);
    foo(a1);
//    printf("%d\n", a);

    return 0;
}

