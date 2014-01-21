/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  17.09.2013 10:48:30
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
#include <functional>
#include <array>
#include <iostream>
// #include </home/primat/projects/cae/main/point/point.h>
#include </home/primat/projects/common/types.h>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>

template <typename... T>
auto zip(const T&... containers) -> boost::iterator_range<boost::zip_iterator<decltype(boost::make_tuple(std::begin(containers)...))>>
{
    auto zip_begin = boost::make_zip_iterator(boost::make_tuple(std::begin(containers)...));
    auto zip_end = boost::make_zip_iterator(boost::make_tuple(std::end(containers)...));
    return boost::make_iterator_range(zip_begin, zip_end);
};

template<typename _Tp, std::size_t _Nm>
struct List
{
    typedef _Tp                         value_type;
    typedef value_type*                 pointer;
    typedef const value_type*                       const_pointer;
    typedef value_type&                             reference;
    typedef const value_type&                       const_reference;
    typedef value_type*                     iterator;
    typedef const value_type*               const_iterator;
    typedef std::size_t                             size_type;
    typedef std::ptrdiff_t                          difference_type;
    typedef std::reverse_iterator<iterator>         reverse_iterator;
    typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;

    // Support for zero-sized Lists mandatory.
    value_type _M_instance[_Nm ? _Nm : 1];

    // No explicit construct/copy/destroy for aggregate type.

    // DR 776.
    void
        fill(const value_type& __u)
        { std::fill_n(begin(), size(), __u); }

    void
        swap(List& __other)
        noexcept(noexcept(swap(std::declval<_Tp&>(), std::declval<_Tp&>())))
        { std::swap_ranges(begin(), end(), __other.begin()); }

    // Iterators.
    iterator
        begin() noexcept
        { return iterator(data()); }

    const_iterator
        begin() const noexcept
        { return const_iterator(data()); }

    iterator
        end() noexcept
        { return iterator(data() + _Nm); }

    const_iterator
        end() const noexcept
        { return const_iterator(data() + _Nm); }

    reverse_iterator 
        rbegin() noexcept
        { return reverse_iterator(end()); }

    const_reverse_iterator 
        rbegin() const noexcept
        { return const_reverse_iterator(end()); }

    reverse_iterator 
        rend() noexcept
        { return reverse_iterator(begin()); }

    const_reverse_iterator 
        rend() const noexcept
        { return const_reverse_iterator(begin()); }

    const_iterator
        cbegin() const noexcept
        { return const_iterator(std::__addressof(_M_instance[0])); }

    const_iterator
        cend() const noexcept
        { return const_iterator(std::__addressof(_M_instance[_Nm])); }

    const_reverse_iterator 
        crbegin() const noexcept
        { return const_reverse_iterator(end()); }

    const_reverse_iterator 
        crend() const noexcept
        { return const_reverse_iterator(begin()); }

    // Capacity.
    constexpr size_type 
        size() const noexcept { return _Nm; }

    constexpr size_type 
        max_size() const noexcept { return _Nm; }

    constexpr bool 
        empty() const noexcept { return size() == 0; }

    // Element access.
    reference
        operator[](size_type __n)
        { return _M_instance[__n]; }

    constexpr const_reference
        operator[](size_type __n) const noexcept
        { return _M_instance[__n]; }

    reference
        at(size_type __n)
        {
            if (__n >= _Nm)
                std::__throw_out_of_range(__N("List::at"));
            return _M_instance[__n];
        }

    constexpr const_reference
        at(size_type __n) const
        {
            // Result of conditional expression must be an lvalue so use
            // boolean ? lvalue : (throw-expr, lvalue)
            return __n < _Nm ? _M_instance[__n]
                : (std::__throw_out_of_range(__N("List::at")), _M_instance[0]);
        }

    reference 
        front()
        { return *begin(); }

    const_reference 
        front() const
        { return *begin(); }

    reference 
        back()
        { return _Nm ? *(end() - 1) : *end(); }

    const_reference 
        back() const
        { return _Nm ? *(end() - 1) : *end(); }

    pointer
        data() noexcept
        { return std::__addressof(_M_instance[0]); }

    const_pointer
        data() const noexcept
        { return std::__addressof(_M_instance[0]); }

    void each (std::function<void(_Tp)> f) {for (_Tp i : _M_instance) f(i);};
};

// List comparisons.
template<typename _Tp, std::size_t _Nm>
    inline bool 
operator==(const List<_Tp, _Nm>& __one, const List<_Tp, _Nm>& __two)
{ return std::equal(__one.begin(), __one.end(), __two.begin()); }

template<typename _Tp, std::size_t _Nm>
    inline bool
operator!=(const List<_Tp, _Nm>& __one, const List<_Tp, _Nm>& __two)
{ return !(__one == __two); }

template<typename _Tp, std::size_t _Nm>
    inline bool
operator<(const List<_Tp, _Nm>& __a, const List<_Tp, _Nm>& __b)
{ 
    return std::lexicographical_compare(__a.begin(), __a.end(),
            __b.begin(), __b.end()); 
}

template<typename _Tp, std::size_t _Nm>
    inline bool
operator>(const List<_Tp, _Nm>& __one, const List<_Tp, _Nm>& __two)
{ return __two < __one; }

template<typename _Tp, std::size_t _Nm>
    inline bool
operator<=(const List<_Tp, _Nm>& __one, const List<_Tp, _Nm>& __two)
{ return !(__one > __two); }

template<typename _Tp, std::size_t _Nm>
    inline bool
operator>=(const List<_Tp, _Nm>& __one, const List<_Tp, _Nm>& __two)
{ return !(__one < __two); }

// Specialized algorithms.
template<typename _Tp, std::size_t _Nm>
    inline void
    swap(List<_Tp, _Nm>& __one, List<_Tp, _Nm>& __two)
noexcept(noexcept(__one.swap(__two)))
{ __one.swap(__two); }

// // Tuple interface to class template List.
// 
// /// tuple_size
// template<typename _Tp> 
// class tuple_size;
// 
// template<typename _Tp, std::size_t _Nm>
// struct tuple_size<List<_Tp, _Nm>>
// : public integral_constant<std::size_t, _Nm> { };
// 
// /// tuple_element
// template<std::size_t _Int, typename _Tp>
// class tuple_element;
// 
// template<std::size_t _Int, typename _Tp, std::size_t _Nm>
// struct tuple_element<_Int, List<_Tp, _Nm> >
// { typedef _Tp type; };
// 
// template<std::size_t _Int, typename _Tp, std::size_t _Nm>
// constexpr _Tp&
// get(List<_Tp, _Nm>& __arr) noexcept
// { return __arr._M_instance[_Int]; }
// 
// template<std::size_t _Int, typename _Tp, std::size_t _Nm>
// constexpr _Tp&&
// get(List<_Tp, _Nm>&& __arr) noexcept
// { return std::move(get<_Int>(__arr)); }
// 
// template<std::size_t _Int, typename _Tp, std::size_t _Nm>
// constexpr const _Tp&
// get(const List<_Tp, _Nm>& __arr) noexcept
// { return __arr._M_instance[_Int]; }

// template <typename T, size_t size>
// struct List : public std::array<T, size>
// {
//     public:
//     // list () {for (size_t i = 0; i < size; ++i) content[i] = T();};
//     // T operator[] (const size_t indx) {return content[indx];};
//     void each (std::function<void(T)> f) {for (T i : content) f(i);};
//     List (std::initializer_list<T> l) {
//         this = l;};
//         // for (size_t i = 0; i < size; ++i) content[i](l[i]);}
//     private:
//     T content[size];
// };

// template <typename T>
// class ilist : public std::initializer_list<T>
// {};
// 
// template<typename T, size_t size>
// void foo (T a[size])
// {
//     puts("jdjdjd");
// };
// 
// template <typename T>
// auto ilist(std::initializer_list<T> l) -> decltype (List<T, l.size()>(l)) 
// {
//     return List<T, l.size()>(l);
// };

template <typename A, typename B>
auto foo(A a, B b) -> decltype (a.size() + b)
{
    return a + b;
};

    // for (1, 10, [] (i) {
    //         puts("hello");});
    // for (size_t i=1; i < 10; ++i)
    //     puts("hello");
    // for (auto i : range(1, 10))
    //     puts("hello");
    // range(1, 10).each([] (i) {
    //         puts("hello");});

class range
{
    public:
        range (size_t end) : iter(0), iter_end(end) {};
        range (size_t begin, size_t end) : iter(begin), iter_end(end) {};
        class Iter
        {
            public:
                Iter (size_t val) : v(val) {};
                void operator ++ () {++v;};
                size_t operator * () {return v;};
                bool operator != (Iter i) {return (v != i.v);};
                // operator size_t() {return v;};

                size_t v = 0;
                // size_t end_v = 0;
        } iter;
        Iter iter_end;
        Iter begin() {return iter;};
        Iter end() {return iter_end;};
    private:
        // range () {};
};

// typedef auto A;
using R = range;
using st = size_t;

class A
{
    public:
        A() : con(0) {};
        A& operator=( const A& rhs )
        {
            A temp( rhs );
            Swap( temp );    // non-throwing
            return *this;
        };
        int con;

    private:
        void Swap( A& rhs ) throw () {con = rhs.con;};
};

class B
{
    public:
        B(){};
        int& operator[](int indx){return i[indx];};
        int i[10];
};

// void test_point(double x1, double y1, double x2, double y2, bool answer)
// {
//     prmt::Point<2> p1(x1, y1), p2(x2, y2);
//     if ((p1 == p2) == answer)
//         printf("(%f, %f) (%f, %f) %d\n", x1, y1, x2, y2, p1 == p2);
//     else
//         printf("(%f, %f) (%f, %f) %d ERROR!\n", x1, y1, x2, y2, p1 == p2);
// };

#include "test.h.gch" 

int main()
{
    arr<int, 3> a = {1,2,3};
    arr<int, 3> b = {4,5,6};
    for (auto i : zip(a,b))
    {
        int i1, i2;
        boost::tie(i1,i2) = i;
        printf("%d %d\n", i1, i2);
    };
    // test_point(1.0, 1.0, 2.0, 2.0, false);
    // test_point(2.0, 1.0, 2.0, 2.0, false);
    // test_point(2.0, 2.0, 2.0, 2.0, true);
    // test_point(2.0, 2.0, 2.1, 2.0, false);
    // test_point(2.1, 2.0, 2.1, 2.0, true);
    // test_point(2.11, 2.0, 2.1, 2.0, false);
    // test_point(1.001, 1.0, 1.0, 1.0, false);
    // test_point(1.0+1e-10, 1.0, 1.0, 1.0, false);
    // test_point(1.0+1e-11, 1.0, 1.0, 1.0, false);
    // printf("%d\n", printf("%d\n", 56));
    // if (false and puts("1")) puts("2"); //lazy
    // B b;
    // b.i[5] = 5;
    // printf("%d\n", b[5]);
    // b[5] = 10;
    // printf("%d\n", b[5]);
    // int a[3] = {3,2,1};
    // std::array<int, 3> b = {5,6,7};
    // // auto foo = std::begin({1,2,3});
    // auto foo = std::begin(b);
    // printf("%d\n", *(++foo));
    // auto r = range(3);
    // auto b = std::begin(r);
    // auto e = std::end(r);
    // ++b;
    // ++b;
    // ++b;
    // printf("%ld %ld %ld %d %ld\n", b.v, range(3).begin().v, *b, (b != e), *e);
    // for(auto i : range(1, 40))
    // {
    //     printf("%ld\n", i);
    // };
    // for(st i: R(1,10))
    // {
    //     printf("%ld\n", i);
    // };
    // A a,b;
    // a.con = 10;
    // b.con = 20;
    // printf("%d %d\n", a.con, b.con);
    // a = b;
    // printf("%d\n", a.con);
    // foo(10, 10.0);
    // foo1(10);
    // foo<int, 3>({1,2,3});
    // ilist<int>({1,2,3});
    // std::initializer_list<int>{1,2,3};
    // std::array({1,2,3});
    // std::array<int, 3> {1,2,3};
    // List<int, 3>{1,2,3}.each([] (int i) {printf("%d\n", i);});
    // List<int, 3>{1,2,3}.each([] (int i) {std::cout << i;});
    // List<int, 3> l;
    // l.each([] (int i) {std::cout << i;});
    // for(auto i : l) std::cout << i;
    // List<int, 10>().each([] (int i) {printf("%d\n", i);});
    // List<float, 10>().each([] (int i) {printf("%f\n", i);});
    // printf("%d\n", l[2]);
    return 1;
}
