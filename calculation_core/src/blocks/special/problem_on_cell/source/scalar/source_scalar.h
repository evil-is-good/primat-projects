#ifndef SOURCE_SCALAR_ON_CELL
#define SOURCE_SCALAR_ON_CELL
 
#include "../../../../general/source/scalar/source_scalar.h"
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/base/quadrature_lib.h>

namespace OnCell
{
//! Элемент вектора праой части уравнения в МКЭ для задачи на ячейке, скалярный случай
/*!
 * Правая часть уравнений на ячейке имеет вид:
 \f[
 \sum_{n=0}^{2}\frac{\partial}{\partial x_n}\lambda_{x_n\nu}; \nu \in \{0, 1, 2\}
 \f]
 где \f$\lambda\f$ - коэфффициент физических свойств материалов входящих 
 в состав композита (например тепропроводность) 
 После интегрированиия по ячейке
 \f[
 \sum_{n=0}^{2}\int_{cell}\frac{\partial}{\partial x_n}\lambda_{x_n\nu} \phi_i; \nu \in \{0, 1, 2\}
 \f]
 Так как производная от прерывной функции бесконечна, то интегрируем по частям чтобы избавится 
 от сингулярности.
 \f[
 \sum_{n=0}^{2}\left(\int_{cell}\frac{\partial}{\partial x_n}(\lambda_{x_n\nu} \phi_i) - \int_{cell}\lambda_{x_n\nu} \frac{\partial}{\partial x_n}\phi_i\right); 
 \nu \in \{0, 1, 2\}
 \f]
 Из-за цикличности границ ячейки первый интеграл равен нулю.
 \f[
 - \sum_{n=0}^{2}\int_{cell}\lambda_{x_n\nu}(q) \frac{\partial}{\partial x_n}\phi_i(q); 
 \nu \in \{0, 1, 2\}
 \f]
  Интеграл расчитывается в квадратурах.
  \f[
  - \sum_{q=0}^{numq} \sum_{n=0}^{2}\int_{cell}\lambda_{x_n\nu}(q) \frac{\partial}{\partial x_n}\phi_i(q) J(q)W(q); \nu \in \{0, 1, 2\}
  \f]
*/
    template <u8 dim>
        class SourceScalar : public ::SourceScalar<dim>
    {
        public:
            SourceScalar (const arr<vec<dbl>, dim> &coefficient, const dealii::FiniteElement<dim> &fe);

            virtual dbl operator() (cst i) override;

            arr<vec<dbl>, dim> coef;
    };

    template <u8 dim>
        SourceScalar<dim>::SourceScalar (
                const arr<vec<dbl>, dim> &coefficient,
                const dealii::FiniteElement<dim> &fe) :
            ::SourceScalar<dim>(fe)
    {
        for (st i = 0; i < dim; ++i)
        {
            for (st j = 0; j < coefficient[i].size(); ++j)
            {
                coef[i] .push_back (coefficient[i][j]);
            };
        };
            // coef(coefficient);
    };

    template <u8 dim>
        dbl SourceScalar<dim>::operator () (cst i)
        {
            const uint8_t num_quad_points = this->quadrature_formula.size();

            dbl res = 0.0;

            for (st q_point = 0; q_point < num_quad_points; ++q_point)
            {
                for(st ort = 0; ort < dim; ++ort)
                {
                    res += 
                        -this->fe_values.shape_grad (i, q_point)[ort] *
                        coef[ort][this->material_id] *
                        this->fe_values.JxW(q_point);
                };
            };

            return res;
        };
};

#endif
