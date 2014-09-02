#ifndef PRMT_TYPES

#define PRMT_TYPES

#include <stdint.h>

typedef int8_t i8; 
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8; 
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef const int8_t ci8;
typedef const int16_t ci16;
typedef const int32_t ci32;
typedef const int64_t ci64;
typedef const uint8_t cu8;
typedef const uint16_t cu16;
typedef const uint32_t cu32;
typedef const uint64_t cu64;

typedef size_t st;
typedef const size_t cst;

typedef float flt;
typedef double dbl;
typedef long double ldbl;

typedef const float cflt;
typedef const double cdbl;
typedef const long double  cldbl;

#include <vector>
#include <array>
#include <functional>

template<typename T>
using vec = std::vector<T>;

template<typename T, size_t dim>
using arr = std::array<T, dim>;

using str = std::string;
//str operator "" _s(const char* s, size_t size)
//{
//    return str(s);
//};

template<typename Signature>
using lmbd = std::function<Signature>;

#endif
