#ifndef CALCULATION_META_COEFFICIENTS_ON_CELL
#define CALCULATION_META_COEFFICIENTS_ON_CELL

#include "../../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"
#include "../../../general/additional_tools/types/types.h"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>

namespace OnCell
{
    //! Расчет среднего коэффициентов
    template<u8 dim>
    dbl calculate_area_of_domain(
            const dealii::DoFHandler<dim> &dof_handler)
    {
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
                    area_of_domain += fe_values.JxW(q_point);
            };
        };
        return  area_of_domain;
    };

    //! Расчет среднего коэффициентов
    template<u8 dim>
    ATools::SecondOrderTensor calculate_mean_coefficients(
            const dealii::DoFHandler<dim> &dof_handler,
            const vec<ATools::SecondOrderTensor> &coef)
    {
        ATools::SecondOrderTensor mean_coefficient;
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
                            coef[cell->material_id()][i][j] *
                            fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < dim; ++i)
                for (st j = 0; j < dim; ++j)
                    mean_coefficient[i][j] /= area_of_domain;
        };
        return  mean_coefficient;
    };

    //! Расчет среднего коэффициентов
    template<u8 dim>
    ATools::FourthOrderTensor calculate_mean_coefficients(
            const dealii::DoFHandler<dim> &dof_handler,
            const vec<ATools::FourthOrderTensor> &coef)
    {
        ATools::FourthOrderTensor mean_coefficient;
        for (st i = 0; i < dim; ++i)
            for (st j = 0; j < dim; ++j)
                for (st k = 0; k < dim; ++k)
                    for (st l = 0; l < dim; ++l)
                        mean_coefficient[i][j][k][l] = 0.0;
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
                            for (st k = 0; k < dim; ++k)
                                for (st l = 0; l < dim; ++l)
                                    mean_coefficient[i][j][k][l] += 
                                        coef[cell->material_id()][i][j][k][l] *
                                        fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < dim; ++i)
                for (st j = 0; j < dim; ++j)
                    for (st k = 0; k < dim; ++k)
                        for (st l = 0; l < dim; ++l)
                            mean_coefficient[i][j][k][l] /= area_of_domain;
        };
        return  mean_coefficient;
    };

    //! Расчет метакоэффициентов, скалярный случай
    template<u8 dim>
    ATools::SecondOrderTensor calculate_meta_coefficients_scalar(
            const dealii::DoFHandler<dim> &dof_handler,
            const arr<dealii::Vector<dbl>, dim> &solution,
            const arr<dealii::Vector<dbl>, dim> &heat_flow,
            const vec<ATools::SecondOrderTensor> &coef)
    {
        ATools::SecondOrderTensor mean_coefficient;
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
                            coef[cell->material_id()][i][j] *
                            fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < dim; ++i)
                for (st j = 0; j < dim; ++j)
                    mean_coefficient[i][j] /= area_of_domain;
        };

        st len_vector_solution = dof_handler.n_dofs();
        ATools::SecondOrderTensor mean_heat_flow;
        ATools::SecondOrderTensor meta_coefficient;

        for (st i = 0; i < dim; ++i)
            for (st j = 0; j < dim; ++j)
            {
                mean_heat_flow[i][j] = 0.0;

                for (st k = 0; k < len_vector_solution; ++k)
                    mean_heat_flow[i][j] += solution[i](k) * (-heat_flow[j](k));

                mean_heat_flow[i][j] /= area_of_domain;

                meta_coefficient[i][j] = mean_coefficient[i][j] + mean_heat_flow[i][j];
            };

        // mean_heat_flow[1][1] = mean_coefficient[1][1];
        // mean_heat_flow[0][1] = meta_coefficient[1][1];
        // return  mean_heat_flow; 
        return  meta_coefficient;
    };

    //! Расчет метакоэффициентов, плоская задача упругости
    template<u8 dim>
    ATools::FourthOrderTensor calculate_meta_coefficients_2d_elastic(
            const dealii::DoFHandler<dim> &dof_handler,
            const SystemsLinearAlgebraicEquations<4> &elastic_slae,
            const SystemsLinearAlgebraicEquations<2> &heat_slae,
            const vec<ATools::FourthOrderTensor> &coef)
    {
        ATools::FourthOrderTensor mean_coefficient;
        for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
                for (st k = 0; k < 3; ++k)
                    for (st l = 0; l < 3; ++l)
                        mean_coefficient[i][j][k][l] = 0.0;
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
                    for (st i = 0; i < 3; ++i)
                        for (st j = 0; j < 3; ++j)
                            for (st k = 0; k < 3; ++k)
                                for (st l = 0; l < 3; ++l)
                                    mean_coefficient[i][j][k][l] += 
                                        coef[cell->material_id()][i][j][k][l] *
                                        fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < 3; ++i)
                for (st j = 0; j < 3; ++j)
                    for (st k = 0; k < 3; ++k)
                        for (st l = 0; l < 3; ++l)
                            mean_coefficient[i][j][k][l] /= area_of_domain;
        };

        st len_vector_solution = dof_handler.n_dofs() + heat_slae.solution[0].size();
        ATools::FourthOrderTensor mean_stress;
        ATools::FourthOrderTensor meta_coefficient;

        auto solution = [&elastic_slae, &heat_slae] (cst i, cst j, cst n) {
            enum {x, y, z};
            cu8 indx[3][3] = {
                {0, 3, x},
                {3, 1, y},
                {x, y, 2}
            };

            cst elastic_size = elastic_slae.solution[0].size();
            cst heat_size    = heat_slae.solution[0].size();

            if ((i != j) and ((i == z) or (j == z)))
            {
                cst n_var = n - elastic_size;
                return n < elastic_size ? 0.0 : heat_slae.solution[indx[i][j]](n_var);
            }
            else
            {
                return n < elastic_size ? elastic_slae.solution[indx[i][j]](n) : 0.0;
            };
        };

        auto stress = [&elastic_slae, &heat_slae] (cst i, cst j, cst n) {
            enum {x, y, z};
            cu8 indx[3][3] = {
                {0, 3, x},
                {3, 1, y},
                {x, y, 2}
            };

            cst elastic_size = elastic_slae.solution[0].size();
            cst heat_size    = heat_slae.solution[0].size();

            if ((i != j) and ((i == z) or (j == z)))
            {
                cst n_var = n - elastic_size;
                return n < elastic_size ? 0.0 : heat_slae.rhsv[indx[i][j]](n_var);
            }
            else
            {
                return n < elastic_size ? elastic_slae.rhsv[indx[i][j]](n) : 0.0;
            };
        };

    for (auto i : {x, y, z}) 
        for (auto j : {x, y, z}) 
            for (auto k : {x, y, z}) 
                for (auto l : {x, y, z}) 
            {
                mean_stress[i][j][k][l] = 0.0;

                for (st m = 0; m < len_vector_solution; ++m)
                    mean_stress[i][j][k][l] += solution(i, j, m) * (-stress(k, l, m));

                mean_stress[i][j][k][l] /= area_of_domain;

                // printf("%ld %ld %ld %ld %f %f %f\n", i, j, k, l,
                //         mean_stress[i][j][k][l], mean_coefficient[i][j][k][l],
                //         mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l]);

                meta_coefficient[i][j][k][l] = 
                    mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l];
            };
    
    // for (st m = 0; m < len_vector_solution; ++m)
    //     printf("%f %f %f\n", 
    //             solution(1, 2, m), -stress(1,2,m), solution(1, 2, m) * (-stress(1, 2, m)));

        return  meta_coefficient;
    };

    //! Расчет метакоэффициентов, обьёмная задача упругости
    template<u8 dim>
    ATools::FourthOrderTensor calculate_meta_coefficients_3d_elastic(
            const dealii::DoFHandler<dim> &dof_handler,
            const SystemsLinearAlgebraicEquations<6> &slae,
            const vec<ATools::FourthOrderTensor> &coef)
    {
        ATools::FourthOrderTensor mean_coefficient;
        for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
                for (st k = 0; k < 3; ++k)
                    for (st l = 0; l < 3; ++l)
                        mean_coefficient[i][j][k][l] = 0.0;
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
                    for (st i = 0; i < 3; ++i)
                        for (st j = 0; j < 3; ++j)
                            for (st k = 0; k < 3; ++k)
                                for (st l = 0; l < 3; ++l)
                                    mean_coefficient[i][j][k][l] += 
                                        coef[cell->material_id()][i][j][k][l] *
                                        fe_values.JxW(q_point);

                for (st q_point = 0; q_point < n_q_points; ++q_point)
                    area_of_domain += fe_values.JxW(q_point);
            };

            for (st i = 0; i < 3; ++i)
                for (st j = 0; j < 3; ++j)
                    for (st k = 0; k < 3; ++k)
                        for (st l = 0; l < 3; ++l)
                            mean_coefficient[i][j][k][l] /= area_of_domain;
        };

        st len_vector_solution = dof_handler.n_dofs();
        ATools::FourthOrderTensor mean_stress;
        ATools::FourthOrderTensor meta_coefficient;

        cu8 indx[3][3] = {
            {0, 3, 4},
            {3, 1, 5},
            {4, 5, 2}
        };

        for (auto i : {x, y, z}) 
            for (auto j : {x, y, z}) 
                for (auto k : {x, y, z}) 
                    for (auto l : {x, y, z}) 
                    {
                        mean_stress[i][j][k][l] = 0.0;

                        for (st m = 0; m < len_vector_solution; ++m)
                            mean_stress[i][j][k][l] += slae.solution[indx[i][j]](m) * (-slae.rhsv[indx[k][l]](m));

                        mean_stress[i][j][k][l] /= area_of_domain;

                        // printf("%ld %ld %ld %ld %f %f %f\n", i, j, k, l,
                        //         mean_stress[i][j][k][l], mean_coefficient[i][j][k][l],
                        //         mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l]);

                        meta_coefficient[i][j][k][l] = 
                            mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l];
                    };

        // for (st m = 0; m < len_vector_solution; ++m)
        //     printf("%f %f %f\n", 
        //             solution(1, 2, m), -stress(1,2,m), solution(1, 2, m) * (-stress(1, 2, m)));

        return  meta_coefficient;
    };

    //! Расчет метакоэффициентов, обьёмная задача упругости, произвольное приближение
    // arr<dbl, 3> calculate_meta_coefficients_3d_elastic_approx(
    //         const arr<i32, 3> approximation,
    //         cst nu_indx,
    //         const dealii::DoFHandler<3> &dof_handler,
    //         const OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> &cell_func)
    // {
    //     ATools::FourthOrderTensor mean_coefficient;
    //     for (st i = 0; i < 3; ++i)
    //         for (st j = 0; j < 3; ++j)
    //             for (st k = 0; k < 3; ++k)
    //                 for (st l = 0; l < 3; ++l)
    //                     mean_coefficient[i][j][k][l] = 0.0;
    //     dbl area_of_domain = 0.0;
    //
    //     {
    //         dealii::QGauss<dim>  quadrature_formula(2);
    //
    //         dealii::FEValues<dim> fe_values (dof_handler.get_fe(), quadrature_formula,
    //                 dealii::update_quadrature_points | dealii::update_JxW_values);
    //
    //         cst n_q_points = quadrature_formula.size();
    //
    //         auto cell = dof_handler.begin_active();
    //         auto endc = dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             fe_values .reinit (cell);
    //
    //             for (st q_point = 0; q_point < n_q_points; ++q_point)
    //                 for (st i = 0; i < 3; ++i)
    //                     for (st j = 0; j < 3; ++j)
    //                         for (st k = 0; k < 3; ++k)
    //                             for (st l = 0; l < 3; ++l)
    //                                 mean_coefficient[i][j][k][l] += 
    //                                     coef[cell->material_id()][i][j][k][l] *
    //                                     fe_values.JxW(q_point);
    //
    //             for (st q_point = 0; q_point < n_q_points; ++q_point)
    //                 area_of_domain += fe_values.JxW(q_point);
    //         };
    //
    //         for (st i = 0; i < 3; ++i)
    //             for (st j = 0; j < 3; ++j)
    //                 for (st k = 0; k < 3; ++k)
    //                     for (st l = 0; l < 3; ++l)
    //                         mean_coefficient[i][j][k][l] /= area_of_domain;
    //     };
    //
    //     st len_vector_solution = dof_handler.n_dofs();
    //     ATools::FourthOrderTensor mean_stress;
    //     ATools::FourthOrderTensor meta_coefficient;
    //
    //     cu8 indx[3][3] = {
    //         {0, 3, 4},
    //         {3, 1, 5},
    //         {4, 5, 2}
    //     };
    //
    //     for (auto i : {x, y, z}) 
    //         for (auto j : {x, y, z}) 
    //             for (auto k : {x, y, z}) 
    //                 for (auto l : {x, y, z}) 
    //                 {
    //                     mean_stress[i][j][k][l] = 0.0;
    //
    //                     for (st m = 0; m < len_vector_solution; ++m)
    //                         mean_stress[i][j][k][l] += slae.solution[indx[i][j]](m) * (-slae.rhsv[indx[k][l]](m));
    //
    //                     mean_stress[i][j][k][l] /= area_of_domain;
    //
    //                     // printf("%ld %ld %ld %ld %f %f %f\n", i, j, k, l,
    //                     //         mean_stress[i][j][k][l], mean_coefficient[i][j][k][l],
    //                     //         mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l]);
    //
    //                     meta_coefficient[i][j][k][l] = 
    //                         mean_coefficient[i][j][k][l] + mean_stress[i][j][k][l];
    //                 };
    //
    //     // for (st m = 0; m < len_vector_solution; ++m)
    //     //     printf("%f %f %f\n", 
    //     //             solution(1, 2, m), -stress(1,2,m), solution(1, 2, m) * (-stress(1, 2, m)));
    //
    //     return  meta_coefficient;
    // };
};

#endif
