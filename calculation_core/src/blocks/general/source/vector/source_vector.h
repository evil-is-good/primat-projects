#ifndef SOURCE_VECTOR
#define SOURCE_VECTOR
 
#include "../scalar/source_scalar.h"

//! Элемент вектора праой части уравнения в МКЭ, векторный случай
/*!
 * Представляет собой интеграл по ячейке:
  \f[
  \int_{cell} f \phi_i 
  \f]
  Интеграл расчитывается в квадратурах.
  \f[
  \sum_{q=0}^{numq} f(q)\phi_i(q)  J(q)W(q)
  \f]
*/
template <u8 dim>
class SourceVector : public SourceInterface<dim>
{
    public:
    typedef std::function<arr<dbl, dim> (const dealii::Point<dim>&)> Func;

    SourceVector (
            const arr<Func, dim> func, const dealii::FiniteElement<dim> &fe);

    virtual dbl operator() (cst i) override;

    virtual void update_on_cell (
        typename dealii::DoFHandler<dim>::active_cell_iterator &cell) override;

    virtual u8 get_dofs_per_cell () override;


    arr<Func, dim>    f; //!< Функция из правой части уравнения.
    SourceScalar<dim> source; 
};

template <u8 dim>
SourceVector<dim>::SourceVector (
        const arr<Func, dim> func, const dealii::FiniteElement<dim> &fe) :
    f(func), source (fe)
{

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
    source.f = f[i % dim];

    return source (i);
};

template <u8 dim>
u8 SourceVector<dim>::get_dofs_per_cell ()
{
    return source.get_dofs_per_cell();
};

#endif
