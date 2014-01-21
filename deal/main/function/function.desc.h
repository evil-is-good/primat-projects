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

#ifndef FEMENIST_FUNCTION_DESC

#define FEMENIST_FUNCTION_DESC

#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <deal.II/base/point.h>

namespace Femenist
{
    namespace AdditionToolsToFunction
    {
        template<class type, uint8_t dim>
        type def_func (const dealii::Point<dim> &p);
    };

    template <class type, uint8_t dim>
    class Function
    {
        public:
            Function ();
            Function (type (*foo) (const dealii::Point<dim> &p));

            void operator= (type (*foo) (const dealii::Point<dim> &p));
            void operator= (const Function<type, dim> &function);
            type operator() (const dealii::Point<dim> &p) const;

        private:
            type (*content) (const dealii::Point<dim> &p);
    };
};


#endif
