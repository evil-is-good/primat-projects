#ifndef SOURCE_VECTOR_ON_CELL
#define SOURCE_VECTOR_ON_CELL
 
#include "../scalar/source_scalar.h"

namespace OnCell
{
    //! Элемент вектора праой части уравнения в МКЭ для задачи на ячейке, векторный случай
    template <u8 dim>
        class SourceVector : public SourceInterface<dim>
    {
        public:
            typedef std::function<arr<dbl, dim> (const dealii::Point<dim>&)> Func;

            SourceVector (
                    const vec<arr<arr<dbl, dim>, dim>> &coefficient, const dealii::FiniteElement<dim> &fe);

            virtual dbl operator() (cst i) override;

            virtual void update_on_cell (
                    typename dealii::DoFHandler<dim>::active_cell_iterator &cell) override;

            virtual u8 get_dofs_per_cell () override;


            vec<arr<arr<dbl, dim>, dim>> coef;
            ::OnCell::SourceScalar<dim> source; 
    };

    template <u8 dim>
        SourceVector<dim>::SourceVector (
                const vec<arr<arr<dbl, dim>, dim>> &coefficient, const dealii::FiniteElement<dim> &fe) :
            source (fe)
    {
        coef .resize (coefficient.size());
        for (st i = 0; i < coefficient.size(); ++i)
        {
            for (st j = 0; j < dim; ++j)
            {
                for (st k = 0; k < dim; ++k)
                {
                    coef[i][j][k] = coefficient[i][j][k];
                };
            };
        };
    };

    template <u8 dim>
        void SourceVector<dim>::update_on_cell (
                typename dealii::DoFHandler<dim>::active_cell_iterator &cell)
        {
            source .update_on_cell (cell);
        };

    template <u8 dim>
        dbl SourceVector<dim>::operator () (cst i)
        {
            source.coef .resize (coef.size());
            for (st m = 0; m < coef.size(); ++m)
            {
                for (st n = 0; n < dim; ++n)
                {
                    source.coef[m][n] = coef[m][i % dim][n];
                };
            };

            return source (i);
        };

    template <u8 dim>
        u8 SourceVector<dim>::get_dofs_per_cell ()
        {
            return source.get_dofs_per_cell();
        };
};
#endif
