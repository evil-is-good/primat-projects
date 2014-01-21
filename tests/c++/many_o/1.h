/*
 * =====================================================================================
 *
 *       Filename:  1.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.07.2013 16:10:21
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef hhh
int foo (int i);

template <typename T>
class cfoo
{
    T i;
    public:
    cfoo () : i(10) {};
    T operator () ();
};
#define hhh
#endif
