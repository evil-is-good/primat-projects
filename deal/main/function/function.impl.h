/*
 * =====================================================================================
 *
 *       Filename:  function.desc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05.09.2012 13:50:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef FEMENIST_FUNCTION_IMPL

#define FEMENIST_FUNCTION_IMPL

//#include "./function.desc.h"

namespace Femenist
{
    namespace AdditionToolsToFunction
    {
        template<class type, uint8_t dim>
        type def_func (const dealii::Point<dim> &p)
        {
            printf("In Femenist::Function using default function\n");
            type res;
            return res;
        };
    };

    template<class type, uint8_t dim>
    Function<type, dim>::Function ()
    {
        content = AdditionToolsToFunction::def_func<type, dim>;
    };

    template<class type, uint8_t dim>
    Function<type, dim>::Function (type (*foo) (const dealii::Point<dim> &p))
    {
        content = &foo;
    };

    template<class type, uint8_t dim>
    void Function<type, dim>::operator= (type (*foo) (const dealii::Point<dim> &p))
    {
        content = foo;
    };

    template<class type, uint8_t dim>
    void Function<type, dim>::operator= (const Function<type, dim> &function)
    {
        this->content = function.content;
    };

    template<class type, uint8_t dim>
    inline type Function<type, dim>::operator() (const dealii::Point<dim> &p) const
    {
        return content(p);
    };
};

#endif
