/*
 * =====================================================================================
 *
 *       Filename:  2.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10.12.2013 14:28:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>

#include <stdlib.h>
#include <stdio.h>
#include <vector>


int main ()
{
    int n = 0;
    // std::vector<double> *v = new std::vector<double>(1000000);
    std::vector<double> v(1000000);
    scanf("%d",&n);
    // delete v;
    v.clear();
    // v.resize(1);
    // v.~vector();
    scanf("%d",&n);
    v.shrink_to_fit();
    scanf("%d",&n);
    // v.~vector();
    // for(size_t i = 0; i < v.size(); ++i)
    //     v[i] = i;
    // scanf("%d",&n);
    // printf("%d\n",n+1);
    return 0;
}
