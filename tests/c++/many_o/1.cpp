/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.07.2013 16:10:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include "1.h"

int foo (int i)
{
    return i * 10;
};

template <typename T>
T cfoo<T>::operator () () {return i;};

template class cfoo<int>;
