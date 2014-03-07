#ifndef SOURCE_SCALAR
#define SOURCE_SCALAR
 
#include "../interface/source_interface.h"
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/base/quadrature_lib.h>

namespace OnCell
{
//! Элемент вектора праой части уравнения в МКЭ, скалярный случай
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
        class SourceScalar : public ::SourceScalar<dim>
    {
        public:
            SourceScalar (const arr<vec<dbl>, dim> &coefficient, const dealii::FiniteElement<dim> &fe);

            virtual dbl operator() (cst i) override;

            const arr<vec<dbl>, dim> coef;
    };

    template <u8 dim>
        SourceScalar<dim>::SourceScalar (
                const arr<vec<dbl>, dim> &coefficient,
                const dealii::FiniteElement<dim> &fe) :
            coef(coefficient),
            quadrature_formula (2),
            fe_values (fe, quadrature_formula,
                    dealii::update_values | dealii::update_quadrature_points | 
                    dealii::update_JxW_values),
            dofs_per_cell (fe.dofs_per_cell),
            num_quad_points (quadrature_formula.size())
    {

    };

    template <u8 dim>
        dbl SourceScalar<dim>::operator () (cst i)
        {
            const uint8_t num_quad_points = quadrature_formula.size();

            dbl res = 0.0;

            for (st q_point = 0; q_point < num_quad_points; ++q_point)
            {
                for(st ort = 0; ort < dim; ++ort)
                {
                    res += 
                        -fe_values.shape_grad (i, q_point)[ort] *
                        coef[ort][material_id] *
                        fe_values.JxW(q_point);
                };
            };

            return res;
        };
};

#endif
