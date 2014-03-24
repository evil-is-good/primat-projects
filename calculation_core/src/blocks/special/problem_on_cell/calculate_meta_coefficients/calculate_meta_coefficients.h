#ifndef CALCULATION_META_COEFFICIENTS_ON_CELL
#define CALCULATION_META_COEFFICIENTS_ON_CELL

#include "../../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>

namespace OnCell
{
    //! Расчет метакоэффициентов, скалярный случай
    template<u8 dim>
    arr<arr<dbl, dim>, dim> calculate_meta_coefficients_scalar(
            const dealii::DoFHandler<dim> &dof_handler,
            const arr<dealii::Vector<dbl>, dim> &solution,
            const arr<dealii::Vector<dbl>, dim> &heat_flow,
            const arr<arr<vec<dbl>, dim>, dim> &coef)
    {
        arr<arr<dbl, dim>, dim> mean_coefficient;
        for (st i = 0; i < dim; ++i)
            for (st j = 0; j < dim; ++j)
                mean_coefficient[i][j] = 0.0;
        dbl area_of_domain = 0.0;

        {
            dealii::QGauss<dim>  quadrature_formula(2);

            dealii::FEValues<dim> fe_values (dof_handler.get_fe(), quadrature_formula,
                    dealii::update_quadrature_points | dealii::update_JxW_values);

            cst n_q_points = quadrature_formula.size();


            auto cell = dof_handler.begin_active();
            auto endc = dof_handler.end();
            for (; cell != endc; ++cell)
            {
                fe_values .reinit (cell);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    for (st i = 0; i < dim; ++i)
                        for (st j = 0; j < dim; ++j)
                        mean_coefficient[i][j] += 
                            coef[i][j][cell->material_id()] *
                            fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < dim; ++i)
                for (st j = 0; j < dim; ++j)
                    mean_coefficient[i][j] /= area_of_domain;
        };

        st len_vector_solution = dof_handler.n_dofs();
        arr<arr<dbl, dim>, dim> mean_heat_flow;
        arr<arr<dbl, dim>, dim> meta_coefficient;

        for (st i = 0; i < dim; ++i)
            for (st j = 0; j < dim; ++j)
            {
                mean_heat_flow[i][j] = 0.0;

                for (st k = 0; k < len_vector_solution; ++k)
                    mean_heat_flow[i][j] += solution[i](k) * (-heat_flow[j](k));

                mean_heat_flow[i][j] /= area_of_domain;

                meta_coefficient[i][j] = mean_coefficient[i][j] + mean_heat_flow[i][j];
            };

        return  meta_coefficient;
    };
};

#endif
