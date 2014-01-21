/*
 * =====================================================================================
 *
 *       Filename:  test.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  27.08.2012 15:36:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */


#include <stdint.h>
#include <stdlib.h>


    template<uint8_t dim, bool type_space>
    struct IndexAndCoor 
    {
    };

    template<bool type_space>
    struct IndexAndCoor<1, type_space> 
    {
        size_t index; // dim - num sides
//        double coor [dim];
    };



    template<bool type_space>
    struct IndexAndCoor<2, type_space> 
    {
        IndexAndCoor();
        ~IndexAndCoor();
        Report set_size(size_t size);
        size_t* index[2][type_space + 1]; // dim - num sides
        double* coor [2];
    };
