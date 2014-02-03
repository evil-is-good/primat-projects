#ifndef heat_conduction_problem_def
#define heat_conduction_problem_def 1

#include <fstream>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_out.h>

#include "../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"
#include "../../general/4_points_function/4_points_function.h"

//! Инструменты для решения задачи теплопроводности
namespace HCPTools
{
    //! Задать тензор теплопроводности 
    template<u8 dim>
    void set_thermal_conductivity(
            arr<arr<vec<dbl>, dim>, dim> &tensor, const arr<arr<vec<dbl>, dim>, dim> &coef)
    {
        for (st i = 0; i < dim; ++i)
        {
            for (st j = 0; j < dim; ++j)
            {
                tensor[i][j] .resize (coef[i][j].size());
                for (st k = 0; k < coef[i][j].size(); ++k)
                {
                    tensor[i][j][k] = coef[i][j][k];
                };
            };
        };
    };

    //! Распечатать в файл температуру
    template<u8 dim>
    void print_temperature (const dealii::Vector<dbl> &temperature, 
                            const dealii::DoFHandler<dim> &dof_handler,
                            const str file_name)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (temperature, "temperature");
        data_out.build_patches ();

        auto name = file_name;
        name += ".gpd";

        std::ofstream output (name);
        data_out.write_gnuplot (output);
    }

    //! Распечатать в файл тепловые потоки
    template<u8 dim>
    void print_heat_conductions (const dealii::Vector<dbl> &temperature, 
                                 arr<arr<vec<dbl>, 2>, 2> &coef,
                                 const Domain<2> &domain,
                                 const str file_name)
    {
    };

    template<>
    void print_heat_conductions<2> (const dealii::Vector<dbl> &temperature, 
                                    arr<arr<vec<dbl>, 2>, 2> &coef,
                                    const Domain<2> &domain,
                                    const str file_name)
    {
        cu32 dofs_per_cell = domain.dof_handler.get_fe().dofs_per_cell;

        arr<dealii::Vector<dbl>, 2> grad;
        arr<dealii::Vector<dbl>, 2> hc;
        
        grad[0] .reinit (domain.dof_handler.n_dofs());
        grad[1] .reinit (domain.dof_handler.n_dofs());

        hc[0] .reinit (domain.dof_handler.n_dofs());
        hc[1] .reinit (domain.dof_handler.n_dofs());

        for (
                auto cell = domain.dof_handler.begin_active(); 
                cell     != domain.dof_handler.end(); 
                ++cell
            )
        {
            arr<prmt::Point<2>, 4> points = {
                cell->vertex(0), cell->vertex(1), 
                cell->vertex(2), cell->vertex(3)};
            arr<dbl, 4> values = {
                temperature(cell->vertex_dof_index (0, 0)),
                temperature(cell->vertex_dof_index (1, 0)),
                temperature(cell->vertex_dof_index (2, 0)),
                temperature(cell->vertex_dof_index (3, 0))};

            Scalar4PointsFunc<2> function_on_cell(points, values);

            for (st i = 0; i < dofs_per_cell; ++i)
            {
                auto indx = cell->vertex_dof_index(i, 0);

                grad[0][indx] = function_on_cell.dx(cell->vertex(i));
                grad[1][indx] = function_on_cell.dy(cell->vertex(i));

                hc[0][indx] = 
                    coef[0][0][cell->material_id()] * grad[0][indx] +
                    coef[0][1][cell->material_id()] * grad[1][indx];
                hc[1][indx] = 
                    coef[1][0][cell->material_id()] * grad[0][indx] +
                    coef[1][1][cell->material_id()] * grad[1][indx];
            };
        };

        {
            dealii::DataOut<2> data_out;
            data_out.attach_dof_handler (domain.dof_handler);
            data_out.add_data_vector (hc[0], "hc_dx");
            data_out.add_data_vector (hc[1], "hc_dy");
            data_out.build_patches ();

            auto name = file_name;
            name += ".gpd";

            std::ofstream output (name);
            data_out.write_gnuplot (output);
        };
    }

    //! Распечатать в файл тепловые градиенты
    template<u8 dim>
    void print_heat_gradient (const dealii::Vector<dbl> &temperature, 
                                 arr<arr<vec<dbl>, 2>, 2> &coef,
                                 const Domain<2> &domain,
                                 const str file_name)
    {
    };

    template<>
    void print_heat_gradient<2> (const dealii::Vector<dbl> &temperature, 
                                    arr<arr<vec<dbl>, 2>, 2> &coef,
                                    const Domain<2> &domain,
                                    const str file_name)
    {
        cu32 dofs_per_cell = domain.dof_handler.get_fe().dofs_per_cell;

        arr<dealii::Vector<dbl>, 2> grad;
        
        grad[0] .reinit (domain.dof_handler.n_dofs());
        grad[1] .reinit (domain.dof_handler.n_dofs());

        vec<dbl> gradx;
        vec<prmt::Point<2>> ps;

        vec<st> N(domain.dof_handler.n_dofs());

        for (
                auto cell = domain.dof_handler.begin_active(); 
                cell     != domain.dof_handler.end(); 
                ++cell
            )
        {
            arr<prmt::Point<2>, 4> points = {
                cell->vertex(0), cell->vertex(1), 
                cell->vertex(2), cell->vertex(3)};
            arr<dbl, 4> values = {
                temperature(cell->vertex_dof_index (0, 0)),
                temperature(cell->vertex_dof_index (1, 0)),
                temperature(cell->vertex_dof_index (2, 0)),
                temperature(cell->vertex_dof_index (3, 0))};

            Scalar4PointsFunc<2> function_on_cell(points, values);

            dbl midl_x = (cell->vertex(0)(0) + cell->vertex(1)(0) + cell->vertex(2)(0) + cell->vertex(3)(0)) / 4.0;
            dbl midl_y = (cell->vertex(0)(1) + cell->vertex(1)(1) + cell->vertex(2)(1) + cell->vertex(3)(1)) / 4.0;

            gradx .push_back (function_on_cell.dy(midl_x, midl_y));
            ps .push_back (prmt::Point<2>(midl_x, midl_y));


                // auto indx = cell->vertex_dof_index(0, 0);

                // grad[0][indx] = function_on_cell.dx(cell->vertex(0));
                // grad[1][indx] = function_on_cell.dy(cell->vertex(0));
            for (st i = 0; i < dofs_per_cell; ++i)
            {
                auto indx = cell->vertex_dof_index(i, 0);

                grad[0][indx] += function_on_cell.dx(cell->vertex(i));
                grad[1][indx] += function_on_cell.dy(cell->vertex(i));
                ++(N[indx]); 
            };
        };
        for (st i = 0; i < N.size(); ++i)
        {
            grad[0][i] /= N[i];
            grad[1][i] /= N[i];
        };

        {
            dealii::DataOut<2> data_out;
            data_out.attach_dof_handler (domain.dof_handler);
            data_out.add_data_vector (grad[0], "grad_dx");
            data_out.add_data_vector (grad[1], "drad_dy");
            data_out.build_patches ();

            auto name = file_name;
            name += ".gpd";

            std::ofstream output (name);
            data_out.write_gnuplot (output);
        };
        FILE *F;
        F = fopen("gradx.gpd", "w");
        for (st i = 0; i < ps.size(); ++i)
        {
            fprintf(F, "%lf %lf %lf\n", ps[i].x(), ps[i].y(), gradx[i]);
        };
        fclose(F);
    }
};

#endif
