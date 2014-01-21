
#ifndef COMMON

#define COMMON

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

    StrongType() : i(0) {};

    explicit operator T () {return i;};
};

//#define DEVELOP

#ifdef DEVELOP

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


#include <vector>
#include <array>

//str operator "" _s(const char* s, size_t size)
//{
//    return str(s);
//};

#endif
