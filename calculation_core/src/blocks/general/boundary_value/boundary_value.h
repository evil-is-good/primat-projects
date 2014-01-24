#ifndef BOUBDARY_VALUE
#define BOUBDARY_VALUE

#include "../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

//! Тип граничного условия
/*!
 * Boundary value type
 */
namespace TBV
{
    cu8 Dirichlet = 0;
    cu8 Neumann   = 1;
};

//! Граничное условие для скалярной задачи
template <u8 dim>
class BoundaryValueScalar
{
    public:
        class FunctionFromDealii : public dealii::Function<dim>
        {
        public:
            typedef std::function<dbl (const dealii::Point<dim>&)> TypeFunc;

            void operator= (const TypeFunc func)
            {
                this->func = func;
            };

            virtual dbl value (const dealii::Point<dim> &p,
                    cu32  component = 0) const override
            {
                return func (p);
            };

            TypeFunc func;
        };

        FunctionFromDealii function;
        st boundary_id;
        u8 boundary_type;
};

//! Граничное условие для векторной задачи
template <u8 dim>
class BoundaryValueVector
{
    public:
        class FunctionFromDealii : public dealii::Function<dim>
        {
        public:
            typedef std::function<arr<dbl, dim> (const dealii::Point<dim>&)> TypeFunc;

            void operator= (const TypeFunc func)
            {
                func = func;
            };

            virtual void vector_value (const dealii::Point<dim> &p,
                    dealii::Vector<dbl> &values) const override
            {
                arr<dbl, dim> res = func(p);
                dealii::Vector<dbl> v(1);
                v[0] = 0.0;
                FOR (i, 0, dim)
                    values = 0.0;//res[i];
            };

            TypeFunc func;
        };

        FunctionFromDealii function;
        st boundary_id;
        u8 boundary_type;
};

#endif
