#ifndef BOUBDARY_VALUE
#define BOUBDARY_VALUE

//! Тип граничного условия
/*!
 * Boundary value type
 */
namespace BVT
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
                function = func;
            };

            virtual dbl value (const dealii::Point<dim> &p,
                    cu32  component = 0) const override
            {
                return function (p);
            };

            TypeFunc function;
        };

        FunctionFromDealii function;
        st boundari_id;
        u8 boundari_type;
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
                function = func;
            };

            virtual void vector_value (const dealii::Point<dim> &p,
                    dealii::Vector<dbl> &values) const override
            {
                auto res = funcion(p);

                FOR (i, 0, dim)
                    values(i) = res[i];
            };

            TypeFunc function;
        };

        FunctionFromDealii function;
        st boundari_id;
        u8 boundari_type;
};

#endif
