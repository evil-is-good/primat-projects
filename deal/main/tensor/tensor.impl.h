/*
 * =====================================================================================
 *
 *       Filename:  tensor.impl.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06.09.2012 16:10:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef FEMENIST_TENSOR_IMPL

#define FEMENIST_TENSOR_IMPL

#include "./tensor.h"

namespace Femenist
{
    template<class type>
    Tensor<type>::Tensor (size_t size)
    {
        content = new type [size];
        t_size  = size;
        len_diagonal = (sqrt((double)(8 * size + 1)) - 1) / 2;
        is_2_rank = abs(trunc(len_diagonal) - len_diagonal) < 0.00000000001;
    };

    template<class type>
    Report Tensor<type>::fill (type *filler, size_t filler_size)
    {
        if (filler_size == t_size)
        {
            for (size i = 0; i < t_size; ++i)
            {
                content[i] = filler[i];
            };

            _return_report_true;
        }
        else
        {
            REPORT_USE(
                    Report report;
                    report.result = false;
                    report.ther_is_a_message = true;
                    report.message = "filler size is not equal tensor size";
                    _return(report);
                   );
        };
    };

    template<class type>
    void Tensor<type>::fill (type filler)
    {
        for (size i = 0; i < t_size; ++i)
        {
            content[i] = filler;
        };
    };

    template<class type>
    type Tensor<type>::operator() (size_t _i)
    {
        if (_i < t_size)
            return content[_i];
        else
        {
            printf("%s i >= tensor size\n", __PRETTY_FUNCTION__);
            return content[0];
        };
    };

    template<class type>
    type Tensor<type>::operator() (size_t _i, size_t _j)
    {
        size_t temp  = t_size;
        size_t dim   = 0;
        float  error = false;

        for (size_t i = 1; i < SIZE_MAX; ++i)
        {
            if (i < temp)
                temp -= i;
            else
                if (i == temp)
                    dim = i;
                else
                    error = true;
        }
        `<...>`
        if (i < t_size)
    };

    template<class type>
    Tensor<type>::~Tensor ()
    {
        delete [] content;
    };
};

#endif
