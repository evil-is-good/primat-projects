#ifndef elastic_problem_def
#define elastic_problem_def 1

#include <fstream>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_out.h>

// #include "../../general/4_points_function/4_points_function.h"

//! Инструменты для решения задачи упругости
namespace EPTools
{
    //! Задать изотропный тэнзор упругости 

    struct set_isotropic_elascity
    {
        cdbl yung;
        cdbl puasson;

        void operator() (ATools::FourthOrderTensor &coef)
        {
            enum {x, y, z};
            for (st i = 0; i < 3; ++i)
            {
                for (st j = 0; j < 3; ++j)
                {
                    for (st k = 0; k < 3; ++k)
                    {
                        for (st l = 0; l < 3; ++l)
                        {
                            coef[i][j][k][l] = 0.0;
                        };
                    };
                };
            };

            const double lambda = 
                (puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
            const double mu     = 
                yung / (2 * (1 + puasson));

            coef[x][x][x][x] = lambda + 2 * mu;
            coef[y][y][y][y] = lambda + 2 * mu;
            coef[z][z][z][z] = lambda + 2 * mu;

            coef[x][x][y][y] = lambda;
            coef[y][y][x][x] = lambda;

            coef[x][y][x][y] = mu;
            coef[y][x][y][x] = mu;
            coef[x][y][y][x] = mu;
            coef[y][x][x][y] = mu;

            coef[x][x][z][z] = lambda;
            coef[z][z][x][x] = lambda;

            coef[x][z][x][z] = mu;
            coef[z][x][z][x] = mu;
            coef[x][z][z][x] = mu;
            coef[z][x][x][z] = mu;

            coef[z][z][y][y] = lambda;
            coef[y][y][z][z] = lambda;

            coef[z][y][z][y] = mu;
            coef[y][z][y][z] = mu;
            coef[z][y][y][z] = mu;
            coef[y][z][z][y] = mu;
        };
    };

    //! Распечатать в файл перемещения
    template<u8 dim>
    void print_move (const dealii::Vector<dbl> &move, 
                            const dealii::DoFHandler<dim> &dof_handler,
                            const str file_name,
                            const dealii::DataOutBase::OutputFormat output_format = dealii::DataOutBase::gnuplot)
    {
            dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        vec<str> solution_names;
        switch (dim)
        {
            case 1:
                solution_names.push_back ("move_x");
                break;
            case 2:
                solution_names.push_back ("move_x");
                solution_names.push_back ("move_y");
                break;
            case 3:
                solution_names.push_back ("move_x");
                solution_names.push_back ("move_y");
                solution_names.push_back ("move_z");
                break;
        };
        data_out.add_data_vector (move, solution_names);
        data_out.build_patches ();

        auto name = file_name;
        // name += ".gpd";

        std::ofstream output (name);
        data_out.write (output, output_format);
    }

    //! Распечатать в файл 2d срез перемещений для 3d случая
    void print_move_slice (const dealii::Vector<dbl> &move, 
                            const dealii::DoFHandler<3> &dof_handler,
                            const str file_name,
                            cst ort,
                            cdbl slice_coor)
    {
        FILE* f_out;
        f_out = fopen (file_name.c_str(), "w");
        for (auto cell = dof_handler.begin_active (); cell != dof_handler.end (); ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<3>::vertices_per_cell; ++i)
            {
                if (std::abs(cell->vertex(i)(ort) - slice_coor) < 1e-10)
                {
                    // dealii::Point<2> point;
                    // for (st j = 0; j < 3; ++j)
                    // {
                    //     if (j != ort)
                    //     {
                    //         point(j) = cell->vetrex(i)(j);
                    //     };
                    // };
                    // cdbl val = temperature(cell->vertex_dof_index(i, 0));
                    // fprintf(f_out, "%f %f %f\n", point(0), point(1), val);
                    fprintf(f_out, "%f %f %f %f %f %f\n",
                            cell->vertex(i)(0),
                            cell->vertex(i)(1),
                            cell->vertex(i)(2),
                            move(cell->vertex_dof_index(i, 0)),
                            move(cell->vertex_dof_index(i, 1)),
                            move(cell->vertex_dof_index(i, 2))
                            );
                };
            };
        };
        fclose(f_out);
    };

    // //! Распечатать в файл тепловые потоки
    // template<u8 dim>
    // void print_heat_conductions (const dealii::Vector<dbl> &temperature, 
    //                              arr<arr<vec<dbl>, 2>, 2> &coef,
    //                              const Domain<2> &domain,
    //                              const str file_name)
    // {
    // };

    // template<>
    // void print_heat_conductions<2> (const dealii::Vector<dbl> &temperature, 
    //                                 arr<arr<vec<dbl>, 2>, 2> &coef,
    //                                 const Domain<2> &domain,
    //                                 const str file_name)
    // {
    //     cu32 dofs_per_cell = domain.dof_handler.get_fe().dofs_per_cell;

    //     arr<dealii::Vector<dbl>, 2> grad;
    //     arr<dealii::Vector<dbl>, 2> hc;
    //     
    //     grad[0] .reinit (domain.dof_handler.n_dofs());
    //     grad[1] .reinit (domain.dof_handler.n_dofs());

    //     hc[0] .reinit (domain.dof_handler.n_dofs());
    //     hc[1] .reinit (domain.dof_handler.n_dofs());

    //     for (
    //             auto cell = domain.dof_handler.begin_active(); 
    //             cell     != domain.dof_handler.end(); 
    //             ++cell
    //         )
    //     {
    //         arr<prmt::Point<2>, 4> points = {
    //             cell->vertex(0), cell->vertex(1), 
    //             cell->vertex(2), cell->vertex(3)};
    //         arr<dbl, 4> values = {
    //             temperature(cell->vertex_dof_index (0, 0)),
    //             temperature(cell->vertex_dof_index (1, 0)),
    //             temperature(cell->vertex_dof_index (2, 0)),
    //             temperature(cell->vertex_dof_index (3, 0))};

    //         Scalar4PointsFunc<2> function_on_cell(points, values);

    //         for (st i = 0; i < dofs_per_cell; ++i)
    //         {
    //             auto indx = cell->vertex_dof_index(i, 0);

    //             grad[0][indx] = function_on_cell.dx(cell->vertex(i));
    //             grad[1][indx] = function_on_cell.dy(cell->vertex(i));

    //             hc[0][indx] = 
    //                 coef[0][0][cell->material_id()] * grad[0][indx] +
    //                 coef[0][1][cell->material_id()] * grad[1][indx];
    //             hc[1][indx] = 
    //                 coef[1][0][cell->material_id()] * grad[0][indx] +
    //                 coef[1][1][cell->material_id()] * grad[1][indx];
    //         };
    //     };

    //     {
    //         dealii::DataOut<2> data_out;
    //         data_out.attach_dof_handler (domain.dof_handler);
    //         data_out.add_data_vector (hc[0], "hc_dx");
    //         data_out.add_data_vector (hc[1], "hc_dy");
    //         data_out.build_patches ();

    //         auto name = file_name;
    //         name += ".gpd";

    //         std::ofstream output (name);
    //         data_out.write_gnuplot (output);
    //     };
    // }

    // //! Распечатать в файл тепловые градиенты
    // template<u8 dim>
    // void print_heat_gradient (const dealii::Vector<dbl> &temperature, 
    //                              arr<arr<vec<dbl>, 2>, 2> &coef,
    //                              const Domain<2> &domain,
    //                              const str file_name)
    // {
    // };

    // template<>
    // void print_heat_gradient<2> (const dealii::Vector<dbl> &temperature, 
    //                                 arr<arr<vec<dbl>, 2>, 2> &coef,
    //                                 const Domain<2> &domain,
    //                                 const str file_name)
    // {
    //     cu32 dofs_per_cell = domain.dof_handler.get_fe().dofs_per_cell;

    //     arr<dealii::Vector<dbl>, 2> grad;
    //     
    //     grad[0] .reinit (domain.dof_handler.n_dofs());
    //     grad[1] .reinit (domain.dof_handler.n_dofs());

    //     dealii::Vector<dbl> T(domain.dof_handler.n_dofs());
    //     vec<bool> flg(domain.dof_handler.n_dofs());
    //     for (st i = 0; i < flg.size(); ++i)
    //     {
    //         flg[i] = false;
    //     };

    //     vec<dbl> gradx;
    //     vec<prmt::Point<2>> ps;

    //     vec<st> N(domain.dof_handler.n_dofs());

    //     FILE *F;
    //     F = fopen("test_iso_3.gpd", "w");
    //     for (
    //             auto cell = domain.dof_handler.begin_active(); 
    //             cell     != domain.dof_handler.end(); 
    //             ++cell
    //         )
    //     {
    //         /* Точки 3 и 2 переставленны местами, потому что в диле у них
    //          * порядок зигзагом, а мне надо по кругу
    //         */
    //         arr<prmt::Point<2>, 4> points = {
    //             prmt::Point<2>(cell->vertex(0)(0), cell->vertex(0)(1)),
    //             prmt::Point<2>(cell->vertex(1)(0), cell->vertex(1)(1)),
    //             prmt::Point<2>(cell->vertex(3)(0), cell->vertex(3)(1)),
    //             prmt::Point<2>(cell->vertex(2)(0), cell->vertex(2)(1))};
    //         arr<dbl, 4> values = {
    //             temperature(cell->vertex_dof_index (0, 0)),
    //             temperature(cell->vertex_dof_index (1, 0)),
    //             temperature(cell->vertex_dof_index (3, 0)),
    //             temperature(cell->vertex_dof_index (2, 0))};

    //         Scalar4PointsFunc<2> function_on_cell(points, values);

    //         dbl midl_x = (cell->vertex(0)(0) + cell->vertex(1)(0) + cell->vertex(2)(0) + cell->vertex(3)(0)) / 4.0;
    //         dbl midl_y = (cell->vertex(0)(1) + cell->vertex(1)(1) + cell->vertex(2)(1) + cell->vertex(3)(1)) / 4.0;

    //         gradx .push_back (function_on_cell.dy(midl_x, midl_y));
    //         ps .push_back (prmt::Point<2>(midl_x, midl_y));
    //         st n = 5;
    //         for (st i = 0; i < n+1; ++i)
    //         {
    //             cdbl s = 1.0 / n * i;
    //             for (st j = 0; j < n+1; ++j)
    //             {
    //                 cdbl r = 1.0 / n * j;
    //                 cdbl x = function_on_cell.a_x +
    //                          function_on_cell.b_x*s +
    //                          function_on_cell.c_x*r +
    //                          function_on_cell.d_x*s*r;
    //                 cdbl y = function_on_cell.a_y +
    //                          function_on_cell.b_y*s +
    //                          function_on_cell.c_y*r +
    //                          function_on_cell.d_y*s*r;
    //                 cdbl f = function_on_cell.f_(s, r);
    //                 cdbl dx = function_on_cell.f_.dx(
    //                           x,
    //                           y,
    //                           function_on_cell.s.dx(x, y)[0],
    //                           function_on_cell.r.dx(x, y)[1]);
    //                 cdbl dy = function_on_cell.f_.dy(
    //                           x,
    //                           y,
    //                           function_on_cell.s.dy(x, y)[0],
    //                           function_on_cell.r.dy(x, y)[1]);
    //                 fprintf(F, "%f %f %f %f %f\n", x, y, f, dx, dy);
    //             };
    //         };
    //         // if (std::abs(function_on_cell.dy(midl_x, midl_y)) > 1e-5)
    //         //     fprintf(F, "%f %f %f %f %f %f %f %f %f\n",
    //         //             cell->vertex(0)(0), cell->vertex(0)(1),
    //         //             cell->vertex(1)(0), cell->vertex(1)(1),
    //         //             cell->vertex(3)(0), cell->vertex(3)(1),
    //         //             cell->vertex(2)(0), cell->vertex(2)(1),
    //         //             function_on_cell.dy(midl_x, midl_y));

    //             // auto indx = cell->vertex_dof_index(0, 0);

    //             // grad[0][indx] = function_on_cell.dx(cell->vertex(0));
    //             // grad[1][indx] = function_on_cell.dy(cell->vertex(0));
    //         for (st i = 0; i < dofs_per_cell; ++i)
    //         {
    //             auto indx = cell->vertex_dof_index(i, 0);

    //             grad[0][indx] += function_on_cell.dx(cell->vertex(i));
    //             grad[1][indx] += function_on_cell.dy(cell->vertex(i));
    //             ++(N[indx]); 

    //             if (!flg[indx])
    //             {
    //             T[indx] = 
    //                 // temperature(cell->vertex_dof_index (i, 0));
    //                 // function_on_cell(cell->vertex(i));
    //                 // function_on_cell(prmt::Point<2>(cell->vertex(i)(0), cell->vertex(i)(1)));
    //                 // cell->vertex(i)(0) * cell->vertex(i)(0);
    //                 function_on_cell(prmt::Point<2>(cell->vertex(i)(0), cell->vertex(i)(1)))-
    //             temperature(cell->vertex_dof_index (i, 0));
    //             // if (std::abs(T[indx]) > 1e-5)
    //             //     fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %d\n",
    //             //             cell->vertex(0)(0), cell->vertex(0)(1),
    //             //             cell->vertex(1)(0), cell->vertex(1)(1),
    //             //             cell->vertex(3)(0), cell->vertex(3)(1),
    //             //             cell->vertex(2)(0), cell->vertex(2)(1),
    //             //             temperature(cell->vertex_dof_index (0, 0)),
    //             //             temperature(cell->vertex_dof_index (1, 0)),
    //             //             temperature(cell->vertex_dof_index (3, 0)),
    //             //             temperature(cell->vertex_dof_index (2, 0)),
    //             //             T[indx],
    //             //             i
    //             //             );
    //             flg[indx] = true;
    //             };
    //         };
    //     };
    //     fclose(F);
    //     for (st i = 0; i < N.size(); ++i)
    //     {
    //         // printf("%d\n", N[i]);
    //         grad[0][i] /= N[i];
    //         grad[1][i] /= N[i];
    //         // T[i] /= N[i];
    //     };

    //     {
    //         dealii::DataOut<2> data_out;
    //         data_out.attach_dof_handler (domain.dof_handler);
    //         data_out.add_data_vector (grad[0], "grad_dx");
    //         data_out.add_data_vector (grad[1], "drad_dy");
    //         data_out.add_data_vector (T, "T");
    //         data_out.build_patches ();

    //         auto name = file_name;
    //         name += ".gpd";

    //         std::ofstream output (name);
    //         data_out.write_gnuplot (output);
    //     };
    //     FILE *F1;
    //     F1 = fopen("gradx.gpd", "w");
    //     for (st i = 0; i < ps.size(); ++i)
    //     {
    //         fprintf(F1, "%lf %lf %lf\n", ps[i].x(), ps[i].y(), gradx[i]);
    //     };
    //     fclose(F1);
    // }
};

#endif
