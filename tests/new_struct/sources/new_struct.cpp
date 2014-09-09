#include <stdlib.h>
#include <stdio.h>
// #include <../../../prmt_sintactic_addition/prmt_sintactic_addition.h>

#include "../../../calculation_core/src/blocks/general/domain/domain.h"
#include "../../../calculation_core/src/blocks/general/laplacian/scalar/laplacian_scalar.h"
#include "../../../calculation_core/src/blocks/general/source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/general/boundary_value/boundary_value.h"
#include "../../../calculation_core/src/blocks/general/assembler/assembler.h"
#include "../../../calculation_core/src/blocks/general/system_linear_algebraic_equations/system_linear_algebraic_equations.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/trivial_prepare_system_equations/trivial_prepare_system_equations.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/apply_boundary_value/scalar/apply_boundary_value_scalar.h"
#include "../../../calculation_core/src/blocks/general/geometric_tools/geometric_tools.h"
#include "../../../calculation_core/src/blocks/special/heat_conduction_problem_tools/heat_conduction_problem_tools.h"

// #include "../../../calculation_core/src/blocks/special/problem_on_cell/domain_looper/domain_looper.h"
// #include "../../../calculation_core/src/blocks/special/problem_on_cell/black_on_white_substituter/black_on_white_substituter.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations/prepare_system_equations.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations_with_cubic_grid/prepare_system_equations_with_cubic_grid.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/system_linear_algebraic_equations/system_linear_algebraic_equations.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/calculate_meta_coefficients/calculate_meta_coefficients.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/assembler/assembler.h"
// #include "../../../calculation_core/src/blocks/special/problem_on_cell/domain_looper_trivial/domain_looper_trivial.h"

#include "../../../calculation_core/src/blocks/general/laplacian/vector/laplacian_vector.h"
#include "../../../calculation_core/src/blocks/general/source/vector/source_vector.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/apply_boundary_value/vector/apply_boundary_value_vector.h"
#include "../../../calculation_core/src/blocks/special/elastic_problem_tools/elastic_problem_tools.h"

#include "../../../calculation_core/src/blocks/special/problem_on_cell/source/vector/source_vector.h"


#include "../../../calculation_core/src/blocks/special/nikola_problem/source/scalar/source_scalar.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_reordering.h>

extern void make_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<st>);

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<prmt::Point<2>>);

void debputs()
{
    static int n = 0;
    printf("DEBUG %d\n", n);
    n++;
};

class Beeline
{
    public:
        Beeline (arr<prmt::Point<2>, 4> nodes, arr<dbl, 4> f_values)
        {
            cdbl x1 = nodes[0].x();
            cdbl x2 = nodes[1].x();
            cdbl x3 = nodes[2].x();
            cdbl x4 = nodes[3].x();

            cdbl y1 = nodes[0].y();
            cdbl y2 = nodes[1].y();
            cdbl y3 = nodes[2].y();
            cdbl y4 = nodes[3].y();

            cdbl f1 = f_values[0]; 
            cdbl f2 = f_values[1];
            cdbl f3 = f_values[2];
            cdbl f4 = f_values[3];

            a=-(x2*y2*x3*y4*f1+y1*x3*x4*y4*f2-y1*x3*x2*y2*f4-x1*y1*x3*y4*f2-x1*y3*x4*y4*f2-x1*f3*x2*y2*y4+x1*x3*y3*f2*y4+x1*y3*x2*y2*f4+y1*f3*x4*x2*y2-y3*x4*x2*y2*f1-y3*y1*x3*x4*f2+x1*y3*x4*y1*f2-x2*y3*x3*y4*f1+x1*y1*f3*y4*x2+y1*y3*x3*x2*f4+x2*y3*x4*y4*f1-y1*f3*y4*x4*x2+x3*y3*x4*y2*f1-x3*f1*y2*x4*y4+y2*x3*x1*y1*f4+y2*x1*f3*y4*x4-y2*x4*x1*y1*f3-y2*x1*y3*x3*f4-x1*y1*y3*x2*f4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        };
        dbl operator () (cdbl x, cdbl y) 
        {
            return a + b * x + c * y + d * x * y;
        };
        dbl dx (cdbl x, cdbl y) 
        {
            return b + d * y;
        };
        dbl dy (cdbl x, cdbl y) 
        {
            return b + d * y;
        };
        dbl operator () (const prmt::Point<2> &p) 
        {
            return operator()(p.x(), p.y());
        };
        dbl dx (const prmt::Point<2> &p) 
        {
            return dx(p.x(), p.y());
        };
        dbl dy (const prmt::Point<2> &p)
        {
            return dy(p.x(), p.y());
        };

        dbl a, b, c, d;
};

template<uint8_t dim>
dealii::Point<dim, double> get_grad (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> solution,
        uint8_t index_vertex)
{
    dealii::Point<dim, double> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    double f1 = solution(cell->vertex_dof_index (0, 0));
    double f2 = solution(cell->vertex_dof_index (1, 0));
    double f3 = solution(cell->vertex_dof_index (2, 0));
    double f4 = solution(cell->vertex_dof_index (3, 0));

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

    grad(0) = b + d * cell->vertex(index_vertex)(1);
    grad(1) = c + d * cell->vertex(index_vertex)(0);

    return grad;
};

template <size_t num_points>
void set_tria(dealii::Triangulation< 2 > &triangulation, 
        const double points[num_points], 
        const size_t material_id[num_points - 1][num_points - 1])
{

    const size_t num_cells = num_points - 1;

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(points[j], points[i]);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            c[i * num_cells + j].vertices[0] = i * num_points + j + 0;
            c[i * num_cells + j].vertices[1] = i * num_points + j + 1;
            c[i * num_cells + j].vertices[2] = i * num_points + j + num_points;
            c[i * num_cells + j].vertices[3] = i * num_points + j + num_points + 1;

            c[i * num_cells + j].material_id = material_id[i][j];
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

// template<size_t dim>
// std::array<double, 3> solved (dealii::Triangulation<dim> &triangulation,
//         const double coef_1, const double coef_2)
// {
//     std::array<std::vector<double>, 3> coef;
// 
//     coef[0] .resize (2);
//     coef[1] .resize (2);
//     coef[2] .resize (2);
// 
//     coef[0][0] = coef_1; coef[0][1] = coef_2;
//     coef[1][0] = coef_1; coef[1][1] = coef_2;
//     coef[2][0] = 0.0;    coef[2][1] = 0.0;
// 
//     class ::HeatConductionProblemOnCell<dim> problem (triangulation, coef);
// 
//     REPORT problem .solved ();
// 
//     problem .print_result ("res_");
// 
//     dbl max_s = 0.0;
//     for (auto i : problem.solution[0])
//         i
// 
// //    printf("%f %f %f\n", problem.meta_coefficient[0],
// //                         problem.meta_coefficient[1],
// //                         problem.meta_coefficient[2]);
// 
// 
// //    {
// //       dealii::Vector<double> 
// //           grad(problem.system_equations.x.size());
// //       {
// //           typename dealii::DoFHandler<2>::active_cell_iterator cell =
// //               problem.domain.dof_handler.begin_active();
// //
// //           typename dealii::DoFHandler<2>::active_cell_iterator endc =
// //               problem.domain.dof_handler.end();
// //
// //           std::vector<uint8_t> 
// //               divider(problem.system_equations.x.size());
// //
// //           for (; cell != endc; ++cell)
// //           {
// //               double tau = 0.0
// //               FOR_I(0, 4)
// //               {
// ////                   for (size_t q_point = 0; q_point < 4; ++q_point)
// ////                   {
// ////                       FOR_J(0, 2)
// ////                       {
// ////                           tau += -fe_values.shape_grad (index_i, q_point)[i] *
// ////                               this->coefficient[i][material_id] *
// ////                               fe_values.JxW(q_point);
// ////                       };
// ////                   };
// //                   grad(cell->vertex_dof_index(i,0)) +=
// //                       get_grad<dim>(cell, problem.solution[0], i)[0];
// //
// //                   divider[cell->vertex_dof_index(i,0)] += 1;
// //               // printf("I %d %f\n", cell->vertex_dof_index(i,0),
// //               //         get_grad<dim>(cell, problem.solution[0], i)[0]);
// //               };
// //               FOR_I(0, 4)
// //                   if (cell->vertex_dof_index(i,0) == 35)
// //                   {
// //                       printf("%d %d %d %d %f\n", 
// //                               cell->vertex_dof_index(0,0),
// //                               cell->vertex_dof_index(1,0),
// //                               cell->vertex_dof_index(2,0),
// //                               cell->vertex_dof_index(3,0),
// //                               get_grad<dim>(cell, problem.solution[0], i)[0]);
// //                       break;
// //                   };
// //           };
// //           FOR_I(0, divider.size())
// //           {
// //               grad(i) /= divider[i];
// //               // printf("A %d %f\n", i,
// //               //         grad(i));
// //               // grad(i)[1] /= divider[i];
// //           };
// //       };
// //       {
// //           dealii::DataOut<dim> data_out;
// //           data_out.attach_dof_handler (problem.domain.dof_handler);
// //
// //          char suffix[3] = {'x', 'y', 'z'};
// //
// //          for (uint8_t i = 0; i < 1; ++i)
// //          {
// //             data_out.add_data_vector (grad, "grad");
// //             data_out.build_patches ();
// //
// //              std::string file_name = "grad_x";
// //              file_name += suffix[i];//i;//boost::lexical_cast<char> (i);
// //              file_name += ".gpd";
// //
// //              std::ofstream output (file_name.data());
// //              data_out.write_gnuplot (output);
// //          };
// //       };
// //    };
// 
// 
//     std::array<double, 3> meta;
//     meta[1] = max_s;
//     meta[0] = problem.meta_coefficient[0];
//     // meta[1] = problem.meta_coefficient[1];
//     meta[2] = problem.meta_coefficient[2];
// 
//     printf("meta %lf %lf %lf\n", meta[0], meta[1], meta[2]);
// 
//     return meta;
// 
// };

template <u8 dim>
void solved_heat_problem_on_cell (
        const dealii::Triangulation<dim> &grid,
        const vec<ATools::SecondOrderTensor> &coef,
        OnCell::SystemsLinearAlgebraicEquations<dim> &slae)
{
    enum {x, y, z};

    Domain<dim> domain;
    domain.grid .copy_triangulation(grid);
    dealii::FE_Q<dim> fe(1);
    domain.dof_init (fe);

    OnCell::BlackOnWhiteSubstituter bows;

    LaplacianScalar<dim> element_matrix (domain.dof_handler.get_fe());

    element_matrix.C .resize(2);
    for (st i = 0; i < coef.size(); ++i)
    {
        element_matrix.C[i][x][x] = coef[i][x][x];
        element_matrix.C[i][x][y] = coef[i][x][y];
        element_matrix.C[i][y][x] = coef[i][y][x];
        element_matrix.C[i][y][y] = coef[i][y][y];
    };

    const bool scalar_type = 0;
    OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);

    OnCell::Assembler::assemble_matrix<dim> (slae.matrix, element_matrix, domain.dof_handler, bows);
    // FILE *F;
    // F = fopen("matrix.gpd", "w");
    // for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
    //     for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
    //         if (slae.matrix.el(i,j))
    //         {
    //             fprintf(F, "%ld %ld %f\n", i, j, slae.matrix(i,j));
    //         };
    // fclose(F);

    FOR(i, 0, dim)
    {
        vec<arr<dbl, 2>> coef_for_rhs(2);
        FOR(j, 0, element_matrix.C.size())
        {
            FOR(k, 0, 2)
            {
                coef_for_rhs[j][k] = element_matrix.C[j][i][k];
            };
        };
        OnCell::SourceScalar<dim> element_rhsv (coef_for_rhs, domain.dof_handler.get_fe());
        OnCell::Assembler::assemble_rhsv<dim> (slae.rhsv[i], element_rhsv, domain.dof_handler, bows);
        {
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler (domain.dof_handler);
            data_out.add_data_vector (slae.rhsv[0], "xb");
            data_out.add_data_vector (slae.rhsv[1], "yb");
            data_out.build_patches ();

            auto name = "b.gpd";

            std::ofstream output (name);
            data_out.write_gnuplot (output);
        };

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution[i],
                slae.rhsv[i]
                ,dealii::PreconditionIdentity()
                );
        FOR(j, 0, slae.solution[i].size())
            slae.solution[i][j] = slae.solution[i][bows.subst (j)];
    };
};

template<size_t size>
std::array<size_t, 2> to2D(const size_t i)//, const size_t j)
{
    std::array<size_t, 2> res;

    switch (size)
    {
        case 6*6:
            {
                if (i < 3)
                {
                    res[0] = i;
                    res[1] = i;
                }
                else
                {
                    switch (i)
                    {
                        case 3: res[0]=1; res[1]=2; break; 
                        case 4: res[0]=2; res[1]=0; break; 
                        case 5: res[0]=0; res[1]=1; break;
                    };
                };
            };
            break;
        case 9*9:
            {
                    res[0] = i / 3;
                    res[1] = i % 3;
            };
            break;
    };

    return res;
};

template<size_t size>
void print_tensor(const ATools::FourthOrderTensor &tensor)
{
    const size_t width = static_cast<size_t>(sqrt(size));

    for (size_t i = 0; i < width; ++i)
    {
        auto ind = to2D<size>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width; ++j)
        {
            auto jnd = to2D<size>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            if (fabs(tensor[im][in][jm][jn]) > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", 
                        tensor[im][in][jm][jn]);
            else
                printf("%f   ", 
                        tensor[im][in][jm][jn]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("\n");
};

void set_hexagon_grid_pure(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{
    double Ro = len_edge;
    double ro = radius;
    double Ri = Ro * (sqrt(3.0) / 2.0);
    double ri = ro * (sqrt(3.0) / 2.0);

    printf("Radius %f %f\n", ro, ri);

    double a[7] = {0.0, Ri-ri, Ri, Ri+ri, 2.0*Ri, ri, 2.0*Ri - ri};
    double b[15] = {
        0.0, ro/2.0, ro, Ro/2, Ro, 1.5*Ro - ro, 1.5*Ro - ro / 2.0,
        1.5*Ro, 1.5*Ro + ro / 2.0, 1.5*Ro + ro, 2.0*Ro, 2.5*Ro, 3.0*Ro-ro,
        3.0*Ro-ro/2.0, 3.0*Ro};


    std::vector<dealii::Point< 2 > > v (30); //30

//    v[0][0]  = a[0]; v[0][1]  = b[0];
//    v[1][0]  = a[1]; v[1][1]  = b[0];
//    v[2][0]  = a[1]; v[2][1]  = b[1];
//    v[3][0]  = a[0]; v[3][1]  = b[1];
//    v[4][0]  = a[4]; v[4][1]  = b[3];
//    v[5][0]  = a[2]; v[5][1]  = b[4];
//    v[6][0]  = a[2]; v[6][1]  = b[10];
//    v[7][0]  = a[0]; v[7][1]  = b[11];
//    v[8][0]  = a[4]; v[8][1]  = b[11];
//    v[9][0]  = a[0]; v[9][1]  = b[14];
//    v[10][0] = a[2]; v[10][1] = b[14];
//    v[11][0] = a[4]; v[11][1] = b[14];
    

    v[0][0]  = a[0]; v[0][1]  = b[0];
    v[1][0]  = a[1]; v[1][1]  = b[0];
    v[2][0]  = a[2]; v[2][1]  = b[0];
    v[3][0]  = a[3]; v[3][1]  = b[0];
    v[4][0]  = a[4]; v[4][1]  = b[0];

    v[5][0]  = a[1]; v[5][1]  = b[1];
    v[6][0]  = a[3]; v[6][1]  = b[1];

    v[7][0]  = a[2]; v[7][1]  = b[2];

    v[8][0]  = a[0]; v[8][1]  = b[3];
    v[9][0]  = a[4]; v[9][1]  = b[3];

    v[10][0] = a[2]; v[10][1] = b[4];

    v[11][0] = a[0]; v[11][1] = b[5];
    v[12][0] = a[4]; v[12][1] = b[5];

    v[13][0] = a[5]; v[13][1] = b[6];
    v[14][0] = a[6]; v[14][1] = b[6];

    v[15][0] = a[5]; v[15][1] = b[8];
    v[16][0] = a[6]; v[16][1] = b[8];

    v[17][0] = a[0]; v[17][1] = b[9];
    v[18][0] = a[4]; v[18][1] = b[9];

    v[19][0] = a[2]; v[19][1] = b[10];

    v[20][0] = a[0]; v[20][1] = b[11];
    v[21][0] = a[4]; v[21][1] = b[11];

//    v[23][0] = a[4]; v[23][1] = b[8];

    v[22][0] = a[2]; v[22][1] = b[12];

    v[23][0] = a[1]; v[23][1] = b[13];
    v[24][0] = a[3]; v[24][1] = b[13];

    v[25][0] = a[0]; v[25][1] = b[14];
    v[26][0] = a[1]; v[26][1] = b[14];
    v[27][0] = a[2]; v[27][1] = b[14];
    v[28][0] = a[3]; v[28][1] = b[14];
    v[29][0] = a[4]; v[29][1] = b[14];
//
////    v[31][0] = a[4]; v[31][1] = b[8];  // 13

    std::vector< dealii::CellData< 2 > > c (20, dealii::CellData<2>()); //20

//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 5;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 5;
//    c[1].vertices[1] = 1;
//    c[1].vertices[2] = 2;
//    c[1].vertices[3] = 4;
//    c[1].material_id = 0;
//
//    c[2].vertices[0] = 3;
//    c[2].vertices[1] = 5;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 5;
//    c[3].vertices[1] = 4;
//    c[3].vertices[2] = 8;
//    c[3].vertices[3] = 6;
//    c[3].material_id = 0;
//
//    c[4].vertices[0] = 7;
//    c[4].vertices[1] = 6;
//    c[4].vertices[2] = 10;
//    c[4].vertices[3] = 9;
//    c[4].material_id = 0;
//
//    c[5].vertices[0] = 6;
//    c[5].vertices[1] = 8;
//    c[5].vertices[2] = 11;
//    c[5].vertices[3] = 10;
//    c[5].material_id = 0;

//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);
//
////    dealii::GridReordering<2,2>::invert_all_cells_of_negative_grid 
////        (v, c);
//    dealii::GridReordering<2>::reorder_cells(c);
//
//    puts("/////////////////////////////////");
//
//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);



//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;

    c[0].vertices[0] = 1;
    c[0].vertices[1] = 5;
    c[0].vertices[2] = 8;
    c[0].vertices[3] = 0;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 7;
    c[1].vertices[3] = 5;
    c[1].material_id = 1;

    c[2].vertices[0] = 2;
    c[2].vertices[1] = 3;
    c[2].vertices[2] = 6;
    c[2].vertices[3] = 7;
    c[2].material_id = 1;

    c[3].vertices[0] = 3;
    c[3].vertices[1] = 4;
    c[3].vertices[2] = 9;
    c[3].vertices[3] = 6;
    c[3].material_id = 0;

    c[4].vertices[0] = 8;
    c[4].vertices[1] = 5;
    c[4].vertices[2] = 7;
    c[4].vertices[3] = 10;
    c[4].material_id = 0;

    c[5].vertices[0] = 7;
    c[5].vertices[1] = 6;
    c[5].vertices[2] = 9;
    c[5].vertices[3] = 10;
    c[5].material_id = 0;

    c[6].vertices[0] = 8;
    c[6].vertices[1] = 10;
    c[6].vertices[2] = 13;
    c[6].vertices[3] = 11;
    c[6].material_id = 0;

    c[7].vertices[0] = 10;
    c[7].vertices[1] = 9;
    c[7].vertices[2] = 12;
    c[7].vertices[3] = 14;
    c[7].material_id = 0;

    c[8].vertices[0] = 11;
    c[8].vertices[1] = 13;
    c[8].vertices[2] = 15;
    c[8].vertices[3] = 17;
    c[8].material_id = 1;

    c[9].vertices[0] = 13;
    c[9].vertices[1] = 10;
    c[9].vertices[2] = 19;
    c[9].vertices[3] = 15;
    c[9].material_id = 0;

    c[10].vertices[0] = 10;
    c[10].vertices[1] = 14;
    c[10].vertices[2] = 16;
    c[10].vertices[3] = 19;
    c[10].material_id = 0;

    c[11].vertices[0] = 14;
    c[11].vertices[1] = 12;
    c[11].vertices[2] = 18;
    c[11].vertices[3] = 16;
    c[11].material_id = 1;

    c[12].vertices[0] = 17;
    c[12].vertices[1] = 15;
    c[12].vertices[2] = 19;
    c[12].vertices[3] = 20;
    c[12].material_id = 0;

//    c[13].vertices[0] = 15;
//    c[13].vertices[1] = 17;
//    c[13].vertices[2] = 31; //// 31
//    c[13].vertices[3] = 19;
//    c[13].material_id = 1;

    c[13].vertices[0] = 16;
    c[13].vertices[1] = 18;
    c[13].vertices[2] = 21;
    c[13].vertices[3] = 19;
    c[13].material_id = 0;

    c[14].vertices[0] = 20;
    c[14].vertices[1] = 19;
    c[14].vertices[2] = 22;
    c[14].vertices[3] = 23;
    c[14].material_id = 0;

    c[15].vertices[0] = 19;
    c[15].vertices[1] = 21;
    c[15].vertices[2] = 24;
    c[15].vertices[3] = 22;
    c[15].material_id = 0;

    c[16].vertices[0] = 20;
    c[16].vertices[1] = 23;
    c[16].vertices[2] = 26;
    c[16].vertices[3] = 25;
    c[16].material_id = 0;

    c[17].vertices[0] = 23;
    c[17].vertices[1] = 22;
    c[17].vertices[2] = 27;
    c[17].vertices[3] = 26;
    c[17].material_id = 1;

    c[18].vertices[0] = 22;
    c[18].vertices[1] = 24;
    c[18].vertices[2] = 28;
    c[18].vertices[3] = 27;
    c[18].material_id = 1;

    c[19].vertices[0] = 24;
    c[19].vertices[1] = 21;
    c[19].vertices[2] = 29;
    c[19].vertices[3] = 28;
    c[19].material_id = 0;

    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    dealii::GridReordering<2>::reorder_cells(c);
    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
//    triangulation .refine_global (n_ref);

    std::ofstream out ("grid-2.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (triangulation , out);
};

ATools::FourthOrderTensor unphysical_to_physicaly (
        ATools::FourthOrderTensor &unphys)
{
    enum {x, y, z};
    ATools::FourthOrderTensor res;

    double A = 
        unphys[x][x][x][x] * unphys[y][y][y][y] * unphys[z][z][z][z] - 
        unphys[y][y][z][z] * unphys[z][z][y][y] * unphys[x][x][x][x] +
        unphys[x][x][y][y] * unphys[y][y][z][z] * unphys[z][z][x][x] - 
        unphys[y][y][x][x] * unphys[x][x][y][y] * unphys[z][z][z][z] - 
        unphys[y][y][y][y] * unphys[x][x][z][z] * unphys[z][z][x][x] +
        unphys[y][y][x][x] * unphys[x][x][z][z] * unphys[z][z][y][y]; 

//    printf("%f %f %f A = %f\n", 
//            unphys[x][x][x][x][0], 
//            unphys[y][y][y][y][0], 
//            unphys[z][z][z][z][0], 
//            A);

    for (uint8_t i = 0; i < 3; ++i)
    {
        int no_1 = (i + 1) % 3;
        int no_2 = (i + 2) % 3;

        for (uint8_t j = 0; j < 3; ++j)
        {
            int k = (j == no_1) ? no_2 : no_1;

            if (i == j)
                res[i][i][j][j] = A;
            else
                res[i][i][j][j] = 
                    (unphys[i][i][j][j] * unphys[k][k][k][k] -
                     unphys[i][i][k][k] * unphys[j][j][k][k]);

            res[i][i][j][j] /= 
                (unphys[no_1][no_1][no_1][no_1] * 
                 unphys[no_2][no_2][no_2][no_2] - 
                 unphys[no_1][no_1][no_2][no_2] * 
                 unphys[no_2][no_2][no_1][no_1]);
        };
    };
        
    return res;

};

void set_circ(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
   puts("1111111111111111111111111111111111111111111");
    triangulation .refine_global (n_refine);
    {
        dealii::Point<2> center (0.5, 0.5);
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 4; ++i)
            {
                midle_p(0) += cell->vertex(i)(0);
                midle_p(1) += cell->vertex(i)(1);
            };
            midle_p(0) /= 4.0;
            midle_p(1) /= 4.0;

           printf("%f %f\n", midle_p(0), midle_p(1));

            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
//                puts("adf");
            }
            else
                cell->set_material_id(0);
        };
    };
};

void set_cylinder(dealii::Triangulation< 3 > &triangulation, 
        const double radius, cst ort, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
   // puts("1111111111111111111111111111111111111111111");
    triangulation .refine_global (n_refine);
    {
        dealii::Point<2> center (0.5, 0.5);
        dealii::Triangulation<3>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 8; ++i)
            {
                st count = 0;
                for (st j = 0; j < 3; ++j)
                {
                    if (j != ort)
                    {
                        midle_p(count) += cell->vertex(i)(j);
                        ++count;
                    };
                };
                // midle_p(0) += cell->vertex(i)(0);
                // midle_p(1) += cell->vertex(i)(1);
                // midle_p(0) += cell->vertex(i)(0);
                // midle_p(1) += cell->vertex(i)(2);
            };
            midle_p(0) /= 8.0;
            midle_p(1) /= 8.0;

           // printf("%f %f\n", midle_p(0), midle_p(1));

            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
//                puts("adf");
            }
            else
                cell->set_material_id(0);
        };
    };
};

void set_ball(dealii::Triangulation< 3 > &triangulation, 
        const double radius, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
   puts("1111111111111111111111111111111111111111111");
    triangulation .refine_global (n_refine);
    {
        dealii::Point<3> center (0.5, 0.5, 0.5);
        dealii::Triangulation<3>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<3> midle_p(0.0, 0.0, 0.0);

            for (size_t i = 0; i < 8; ++i)
            {
                midle_p(0) += cell->vertex(i)(0);
                midle_p(1) += cell->vertex(i)(1);
                midle_p(2) += cell->vertex(i)(2);
            };
            midle_p(0) /= 8.0;
            midle_p(1) /= 8.0;
            midle_p(2) /= 8.0;

           // printf("%f %f\n", midle_p(0), midle_p(1));

            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
//                puts("adf");
            }
            else
                cell->set_material_id(0);
        };
    };
};

void set_ball(dealii::Triangulation< 3 > &triangulation, 
        const dealii::Point<3> center, cdbl radius)
{
    dealii::Triangulation<3>::active_cell_iterator
        cell = triangulation .begin_active(),
             end_cell = triangulation .end();
    for (; cell != end_cell; ++cell)
    {
        dealii::Point<3> midle_p(0.0, 0.0, 0.0);

        for (size_t i = 0; i < 8; ++i)
        {
            midle_p(0) += cell->vertex(i)(0);
            midle_p(1) += cell->vertex(i)(1);
            midle_p(2) += cell->vertex(i)(2);
        };
        midle_p(0) /= 8.0;
        midle_p(1) /= 8.0;
        midle_p(2) /= 8.0;

        // printf("%f %f\n", midle_p(0), midle_p(1));

        if (center.distance(midle_p) < radius)
        {
            cell->set_material_id(1);
                           // puts("adf");
        }
        // else
        //     cell->set_material_id(0);
        // printf("first %ld\n", cell->material_id());
    };
};

void set_cube(dealii::Triangulation< 3 > &triangulation, 
        const dealii::Point<3> center, cdbl half_len)
{
    dealii::Triangulation<3>::active_cell_iterator
        cell = triangulation .begin_active(),
             end_cell = triangulation .end();
    for (; cell != end_cell; ++cell)
    {
        dealii::Point<3> midle_p(0.0, 0.0, 0.0);

        for (size_t i = 0; i < 8; ++i)
        {
            midle_p(0) += cell->vertex(i)(0);
            midle_p(1) += cell->vertex(i)(1);
            midle_p(2) += cell->vertex(i)(2);
        };
        midle_p(0) /= 8.0;
        midle_p(1) /= 8.0;
        midle_p(2) /= 8.0;

        // printf("%f %f\n", midle_p(0), midle_p(1));

        if (
                (midle_p(0) < (center(0) + half_len)) and 
                (midle_p(1) < (center(1) + half_len)) and 
                (midle_p(2) < (center(2) + half_len)) and 
                (midle_p(0) > (center(0) - half_len)) and 
                (midle_p(1) > (center(1) - half_len)) and 
                (midle_p(2) > (center(2) - half_len))) 
        {
            cell->set_material_id(1);
                           // puts("adf");
        }
        // else
        //     cell->set_material_id(0);
        // printf("first %ld\n", cell->material_id());
    };
};

void set_circ_in_hex(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
    cdbl hight = sqrt(3.0);
    dealii::Point<2> p1(0.0, 0.0);
    dealii::Point<2> p2(1.0, hight);
    dealii::GridGenerator ::hyper_rectangle (triangulation, p1, p2);
    triangulation .refine_global (n_refine);
    {
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 4; ++i)
            {
                midle_p(0) += cell->vertex(i)(0);
                midle_p(1) += cell->vertex(i)(1);
            };
            midle_p(0) /= 4.0;
            midle_p(1) /= 4.0;

            cell->set_material_id(0);
            {
                dealii::Point<2> center (0.5, 0.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (0.0, hight / 2.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (1.0, hight / 2.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (0.5, hight);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            // {
            //     dealii::Point<2> center (1.5, 0.0);
            //     if (center.distance(midle_p) < radius)
            //         cell->set_material_id(1);
            // };
            // {
            //     dealii::Point<2> center (1.5, hight);
            //     if (center.distance(midle_p) < radius)
            //         cell->set_material_id(1);
            // };
        };
    };
};

// template<uint8_t dim>
// void print_stress(const dealii::DoFHandler<dim> &dof_handler,
//         const OnCell::SystemsLinearAlgebraicEquations<4> &slae,
//      const vec<ATools::FourthOrderTensor> &E,
//      cdbl meta_E, 
//      cdbl meta_nu_yx,
//      cdbl meta_nu_zx)
// {
//     const int8_t x = 0;
//     const int8_t y = 1;
//     const int8_t z = 2;
// 
//     std::vector<std::array<dealii::Point<2>, 3> > 
//         grad_elastic_field(dof_handler.n_dofs());
// 
//     {
//         typename dealii::DoFHandler<2>::active_cell_iterator cell =
//             dof_handler.begin_active();
// 
//         typename dealii::DoFHandler<2>::active_cell_iterator endc =
//             dof_handler.end();
// 
//         std::vector<std::array<uint8_t, 3> > 
//             divider(dof_handler.n_dofs());
// 
//         for (; cell != endc; ++cell)
//         {
//             size_t mat_id = cell->material_id();
// 
//             FOR_I(0, 4)
//                 FOR_M(0, 3)
//                 {
//                     std::array<dealii::Point<2>, 2> deform =
//                         ::get_grad_elastic<dim> (cell, 
//                                 solution[m], i);  
// 
//                     double sigma_xx = 
//                         E[x][x][x][x][mat_id] * deform[x](x) +
//                         E[x][x][x][y][mat_id] * deform[x](y) +
//                         E[x][x][y][x][mat_id] * deform[y](x) +
//                         E[x][x][y][y][mat_id] * deform[y](y) +
//                         E[x][x][m][m][mat_id];
// 
//                     double sigma_xy = 
//                         E[x][y][x][x][mat_id] * deform[x](x) +
//                         E[x][y][x][y][mat_id] * deform[x](y) +
//                         E[x][y][y][x][mat_id] * deform[y](x) +
//                         E[x][y][y][y][mat_id] * deform[y](y) +
//                         E[x][y][m][m][mat_id];
// 
//                     double sigma_yx = 
//                         E[y][x][x][x][mat_id] * deform[x](x) +
//                         E[y][x][x][y][mat_id] * deform[x](y) +
//                         E[y][x][y][x][mat_id] * deform[y](x) +
//                         E[y][x][y][y][mat_id] * deform[y](y) +
//                         E[y][x][m][m][mat_id];
// 
//                     double sigma_yy = 
//                         E[y][y][x][x][mat_id] * deform[x](x) +
//                         E[y][y][x][y][mat_id] * deform[x](y) +
//                         E[y][y][y][x][mat_id] * deform[y](x) +
//                         E[y][y][y][y][mat_id] * deform[y](y) +
//                         E[y][y][m][m][mat_id];
// 
//                     grad_elastic_field[cell->vertex_dof_index(i,0)][m] += 
//                         dealii::Point<2>(sigma_xx, sigma_xy);
// //                    dealii::Point<2>(
// //                            cell->vertex_dof_index(i,0),
// //                            cell->vertex_dof_index(i,0));
// //                        dealii::Point<2>(
// //                                temp[0](0) * 
// //                                (cell->material_id() ? 2.0 : 1.0),
// //                                (temp[0](1) + temp[1](0)) *
// //                                (cell->material_id() ? 2.0 : 1.0) / 
// //                                (2.0 * (1.0 + 0.2)));
// 
//                     grad_elastic_field[cell->vertex_dof_index(i,1)][m] += 
//                         dealii::Point<2>(sigma_yx, sigma_yy);
//                     // dealii::Point<2>(
//                     //         cell->vertex_dof_index(i,1),
//                     //         cell->vertex_dof_index(i,1));
// //                        dealii::Point<2>(
// //                                (temp[0](1) + temp[1](0)) *
// //                                (cell->material_id() ? 2.0 : 1.0) / 
// //                                (2.0 * (1.0 + 0.2)),
// //                                temp[1](0) * 
// //                                (cell->material_id() ? 2.0 : 1.0));
// 
//                     divider[cell->vertex_dof_index(i,0)][m] += 1;
//                     divider[cell->vertex_dof_index(i,1)][m] += 1;
//                 };
//         };
// 
//         FOR_I(0, divider.size())
//             FOR_M(0, 3)
//             grad_elastic_field[i][m] ;///= divider[i][m];
//         FOR_I(0, divider.size())
//             printf("%d %f\n", grad_elastic_field[i][0](0));
//     };
// 
//     {
//         char suffix[3] = {'x', 'y', 'z'};
// 
//         dealii::Vector<double> solution(problem.system_equations.x.size());
//         dealii::Vector<double> sigma[2];
//         sigma[0].reinit(problem.system_equations.x.size());
//         sigma[1].reinit(problem.system_equations.x.size());
// 
//         double nu[3] = {- 1.0 / meta_E, + meta_nu_yx / meta_E, + meta_nu_zx / meta_E}; 
// 
//         FOR_I(0, 2)
//         {
//             FOR_J(0, problem.system_equations.x.size())
//                 sigma[i](j) = 0.0;
//             double integ = 0.0;
// 
//             FOR_M(0, 3)
//             {
// 
//                 FOR_J(0, problem.system_equations.x.size())
//                 {
//                     solution(j) = grad_elastic_field[j][m](i);
//                     sigma[i](j) += nu[m] * solution(j);
//                     integ += nu[m] * solution(j);
//                 };
// 
//                 //            integ /= grad_heat_field.size();
// 
//                 dealii::DataOut<dim> data_out;
//                 data_out.attach_dof_handler (
//                         problem.domain.dof_handler);
//                 printf("%d\n", solution.size());
// 
//                 data_out.add_data_vector (solution, "x y");
//                 data_out.build_patches ();
// 
//                 std::string file_name = "grad_";
//                 file_name += suffix[i];
//                 file_name += "_";
//                 file_name += suffix[m];
//                 file_name += suffix[m];
//                 file_name += ".gpd";
// 
//                 std::ofstream output (file_name.data());
//                 data_out.write_gnuplot (output);
//             };
//             printf("INTEGRAL = %f\n", integ / problem.system_equations.x.size());
// 
//             dealii::DataOut<dim> data_out;
//             data_out.attach_dof_handler (
//                     problem.domain.dof_handler);
// 
//             data_out.add_data_vector (sigma[i], "x y");
//             data_out.build_patches ();
// 
//             std::string file_name = "sigma_";
//             file_name += suffix[i];
//             file_name += ".gpd";
// 
//             std::ofstream output (file_name.data());
//             data_out.write_gnuplot (output);
//         };
//     
//         {
//             dealii::Vector<double> main_stress(problem.system_equations.x.size());
//             dealii::Vector<double> angle(problem.system_equations.x.size());
// //            cos[1].reinit(problem.system_equations.x.size());
// 
//             std::vector<uint8_t> divider(problem.system_equations.x.size());
// 
// //            double max_main_stress[2] = {0.0};
// //            double min_main_stress[2] = {0.0};
// 
//             typename dealii::DoFHandler<2>::active_cell_iterator cell =
//                 problem.domain.dof_handler.begin_active();
// 
//             typename dealii::DoFHandler<2>::active_cell_iterator endc =
//                 problem.domain.dof_handler.end();
// 
//             for (; cell != endc; ++cell)
//             {
//                 FOR_I(0, 4)
//                 {
//                     double L1 = 
//                         sigma[0](cell->vertex_dof_index(i,0)) +
//                         sigma[1](cell->vertex_dof_index(i,1)); 
//                     double L2 = 
//                         sigma[0](cell->vertex_dof_index(i,0)) *
//                         sigma[1](cell->vertex_dof_index(i,1)) -
//                         sigma[0](cell->vertex_dof_index(i,1)) *
//                         sigma[1](cell->vertex_dof_index(i,0));
// 
// //                    main_stress(cell->vertex_dof_index(i,0)) =
//                         double str1 = 
//                         (L1 - sqrt(L1*L1 - 4.0 * L2)) / 2.0;
// //                        printf("s1=%f\n", str1);
// 
// //                    main_stress(cell->vertex_dof_index(i,1)) =
//                         double str2 = 
//                         (L1 + sqrt(L1*L1 - 4.0 * L2)) / 2.0;
// //                        printf("s2=%f\n", str2);
// //                        printf("%f %f %f\n", 
// //                        sigma[0](cell->vertex_dof_index(i,0)),
// //                        sigma[1](cell->vertex_dof_index(i,1)),
// //                        sigma[0](cell->vertex_dof_index(i,1))
// //                                );
// 
//                     double angl1 = atan(
//                             2.0 * sigma[0](cell->vertex_dof_index(i,1)) / 
//                             (sigma[0](cell->vertex_dof_index(i,0)) - 
//                              sigma[1](cell->vertex_dof_index(i,1)))) / 2.0;
//                     double angl2 = angl1 + 3.14159265359 / 2.0; 
// 
//                     if (str1 < str2)
//                     {
//                         double temp = str1;
//                         str1 = str2;
//                         str2 = temp;
// 
//                         temp = angl1;
//                         angl1 = angl2;
//                         angl2 = temp;
//                     };
// 
//                     main_stress(cell->vertex_dof_index(i,0)) += str1;
//                     main_stress(cell->vertex_dof_index(i,1)) += str2;
// 
//                     angle(cell->vertex_dof_index(i,0)) += angl1;
//                     angle(cell->vertex_dof_index(i,1)) += angl2;
// //                    cos[0](cell->vertex_dof_index(i,0)) +=
// //                        sqrt(1.0 / (1.0 + 
// //                                    pow(str1 - 
// //                                        sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
// //                                    /
// //                                    pow(sigma[1](cell->vertex_dof_index(i,0)), 2.0)));
// //
// //                    cos[0](cell->vertex_dof_index(i,1)) +=
// //                        (str1 - 
// //                         sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
// //                        /
// //                        sigma[1](cell->vertex_dof_index(i,0)) 
// //                        *
// //                        cos[0](cell->vertex_dof_index(i,0));
// //
// //                    cos[1](cell->vertex_dof_index(i,0)) +=
// //                        sqrt(1.0 / (1.0 + 
// //                                    pow(str2 - 
// //                                        sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
// //                                    /
// //                                    pow(sigma[1](cell->vertex_dof_index(i,0)), 2.0)));
// //
// //                    cos[1](cell->vertex_dof_index(i,1)) +=
// //                        (str2 - 
// //                         sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
// //                        /
// //                        sigma[1](cell->vertex_dof_index(i,0)) 
// //                        *
// //                        cos[1](cell->vertex_dof_index(i,0));
// 
//                     divider[cell->vertex_dof_index(i,0)] += 1;
//                     divider[cell->vertex_dof_index(i,1)] += 1;
// 
// //                    if (str1 > max_main_stress[0])
// //                        max_main_stress[0] = str1;
// //                    if (str2 > max_main_stress[1])
// //                        max_main_stress[1] = str2;
// //                    if (str1 < min_main_stress[0])
// //                        min_main_stress[0] = str1;
// //                    if (str2 < max_main_stress[1])
// //                        min_main_stress[1] = str2;
//                 };
//             };
// 
//             FOR_I(0, divider.size())
//             {
//                 main_stress(i) /= divider[i];
//                 angle(i) /= divider[i];
// //                cos[0](i) /= divider[i];
// //                cos[1](i) /= divider[i];
//             };
// 
//             {
//                 dealii::DataOut<dim> data_out;
//                 data_out.attach_dof_handler (
//                         problem.domain.dof_handler);
// 
//                 data_out.add_data_vector (main_stress, "1 2");
//                 data_out.build_patches ();
// 
//                 std::string file_name = "main_stress";
// //                file_name += "_2_";
//                 file_name += std::to_string(name_main_stress);
//                 file_name += ".gpd";
//                 printf("DDDDDDDDD %s\n", file_name.data());
// 
//                 std::ofstream output (file_name.data());
// //            FOR_I(0, divider.size())
// //            out << 
//                 data_out.write_gnuplot (output);
//             };
// 
// //            char suffix[2] = {'1', '2'};
// //            FOR_I(0, 2)
// //            {
//                 dealii::DataOut<dim> data_out;
//                 data_out.attach_dof_handler (
//                         problem.domain.dof_handler);
// 
//                 data_out.add_data_vector (angle, "x y");
//                 data_out.build_patches ();
// 
//                 std::string file_name = "angle";
// //                file_name += suffix[i];
//                 file_name += ".gpd";
// 
//                 std::ofstream output (file_name.data());
//                 data_out.write_gnuplot (output);
// //            };
// //                {
// //                    FILE *F;
// //                    F = fopen("min_max", "w");
// //                    fprintf(F, "%f %f\n", min_main_stress[0], max_main_stress[0]);
// //                    fprintf(F, "%f %f\n", min_main_stress[1], max_main_stress[1]);
// //                    fclose(F);
// //                };
// 
//         };
//     };
// 
//     {
//         char suffix[3] = {'x', 'y', 'z'};
// 
//         dealii::Vector<double> solution(grad_heat_field.size());
// 
//         FOR_I(0, dim)
//         {
//             double integ = 0.0;
// 
//             FOR_J(0, grad_heat_field.size())
//             {
//                 solution(j) = grad_heat_field[j](i);
//                 integ += solution(j);
//             };
// 
// //            integ /= grad_heat_field.size();
//             printf("INTEGRAL = %f\n", integ);
// 
//             dealii::DataOut<dim> data_out;
//             data_out.attach_dof_handler (
//                     problem.problem_of_torsion_rod.domain.dof_handler);
//             printf("%d\n", solution.size());
// 
//             data_out.add_data_vector (
//                     solution
// //                    problem.problem_of_torsion_rod.heat_flow[i]
//                     , "solution");
//             data_out.build_patches ();
// 
//             std::string file_name = "grad_z";
//             file_name += suffix[i];
//             file_name += ".gpd";
// 
//             std::ofstream output (file_name.data());
//             data_out.write_gnuplot (output);
//         };
//     };
// 
// };

#define SUM(I, BEGIN, END, BODY) ({dbl tmp = 0.0; for (st I = BEGIN; I < END; ++I) {tmp += BODY;}; tmp;});


// template <typename Func, typename Func2>
st foo(cst i, lmbd<st(cst)> &&func, lmbd<st(cst)> &&func2)
{
    return func2(func(i));
};

void set_long_rod (dealii::Triangulation<3> &triangulation, cdbl len, cdbl size, cst n_refine)
{
    // std::vector< dealii::Point< 3 > > v (8);
    //
    // v[0] = dealii::Point<3>(0.0, 0.0, 0.0);
    // v[1] = dealii::Point<3>(1.0, 0.0, 0.0);
    // v[2] = dealii::Point<3>(1.0, 0.0, 0.0);
    // v[3] = dealii::Point<3>(1.0, 1.0, 0.0);
    // v[4] = dealii::Point<3>(0.0, 0.0, 1.0);
    // v[5] = dealii::Point<3>(1.0, 0.0, 1.0);
    // v[6] = dealii::Point<3>(1.0, 0.0, 1.0);
    // v[7] = dealii::Point<3>(1.0, 1.0, 1.0);
    //         // v[0] = dealii::Point<2>(0.0, 0.0);
    //         // v[1] = dealii::Point<2>(1.0, 0.0);
    //         // v[2] = dealii::Point<2>(1.0, 1.0);
    //         // v[3] = dealii::Point<2>(0.0, 1.0);
    //
    // std::vector< dealii::CellData< 3 > > c (1, dealii::CellData<3>());
    //
    // c[0].vertices[0] = 0; 
    // c[0].vertices[1] = 1; 
    // c[0].vertices[2] = 2;
    // c[0].vertices[3] = 3;
    // c[0].vertices[4] = 4;
    // c[0].vertices[5] = 5;
    // c[0].vertices[6] = 6;
    // c[0].vertices[7] = 7;
    // c[0].material_id = 0; 
    //
    // dealii::SubCellData b;
    //
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 0;
    //     cell.vertices[1] = 2;
    //     cell.vertices[2] = 4;
    //     cell.vertices[3] = 6;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 1;
    //     cell.vertices[1] = 3;
    //     cell.vertices[2] = 5;
    //     cell.vertices[3] = 7;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 0;
    //     cell.vertices[1] = 1;
    //     cell.vertices[2] = 4;
    //     cell.vertices[3] = 5;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 2;
    //     cell.vertices[1] = 3;
    //     cell.vertices[2] = 6;
    //     cell.vertices[3] = 7;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 0;
    //     cell.vertices[1] = 1;
    //     cell.vertices[2] = 2;
    //     cell.vertices[3] = 3;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // {
    //     dealii::CellData<2> cell;
    //     cell.vertices[0] = 4;
    //     cell.vertices[1] = 5;
    //     cell.vertices[2] = 6;
    //     cell.vertices[3] = 7;
    //     cell.boundary_id = 0;
    //     b.boundary_quads .push_back (cell);
    // };
    // // {
    // //     dealii::CellData<1> cell;
    // //     cell.vertices[0] = 1;
    // //     cell.vertices[1] = 2;
    // //     cell.boundary_id = 2;
    // //     b.boundary_lines .push_back (cell);
    // // };
    // // {
    // //     dealii::CellData<1> cell;
    // //     cell.vertices[0] = 2;
    // //     cell.vertices[1] = 3;
    // //     cell.boundary_id = 1;
    // //     b.boundary_lines .push_back (cell);
    // // };
    // // {
    // //     dealii::CellData<1> cell;
    // //     cell.vertices[0] = 3;
    // //     cell.vertices[1] = 0;
    // //     cell.boundary_id = 2;
    // //     b.boundary_lines .push_back (cell);
    // // };
    // // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 0});
    // // b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 2});
    // // b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
    // // b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 2});
    //
    // dealii::GridReordering<3> ::reorder_cells (c);
    // // triangulation .create_triangulation_compatibility (v, c, b);
    // // triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
    // triangulation .create_triangulation (v, c, dealii::SubCellData());
    // // triangulation .create_triangulation (v, c, b);

    dealii::Point<3> p1(0.0, 0.0, 0.0);
    dealii::Point<3> p2(len, 1.0, 1.0);
    dealii::GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.begin_active()->face(0)->set_boundary_indicator(1);
    triangulation.begin_active()->face(1)->set_boundary_indicator(2);
    triangulation.begin_active()->face(2)->set_boundary_indicator(0);
    triangulation.begin_active()->face(3)->set_boundary_indicator(0);
    triangulation.begin_active()->face(4)->set_boundary_indicator(0);
    triangulation.begin_active()->face(5)->set_boundary_indicator(0);
    triangulation .refine_global(n_refine);
    for (st i = 0; i < 1; ++i)
    {
        dealii::Point<3> center(i * 1.0 + 0.5, 0.5, 0.5);
        // set_cube (triangulation, center, size);
        set_ball (triangulation, center, size);
    };
};

void set_speciment (dealii::Triangulation<3> &triangulation, 
        cdbl size_x, cdbl size_y, cdbl size_z, cdbl size_inclusion, cdbl size_cell,
        arr<st, 6> id_border, cst n_refine)
{

    dealii::Point<3> p1(0.0, 0.0, 0.0);
    dealii::Point<3> p2(size_x, size_y, size_z);
    dealii::GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.begin_active()->face(0)->set_boundary_indicator(id_border[0]);
    triangulation.begin_active()->face(1)->set_boundary_indicator(id_border[1]);
    triangulation.begin_active()->face(2)->set_boundary_indicator(id_border[2]);
    triangulation.begin_active()->face(3)->set_boundary_indicator(id_border[3]);
    triangulation.begin_active()->face(4)->set_boundary_indicator(id_border[4]);
    triangulation.begin_active()->face(5)->set_boundary_indicator(id_border[5]);
    triangulation .refine_global(n_refine);
    st num_cell_x = st(size_x / size_cell);
    st num_cell_y = st(size_y / size_cell);
    st num_cell_z = st(size_z / size_cell);
    printf("%ld, %ld %ld\n", num_cell_x, num_cell_y, num_cell_z);
    for (st i = 0; i < num_cell_x; ++i)
    {
        for (st j = 0; j < num_cell_y; ++j)
        {
            for (st k = 0; k < num_cell_z; ++k)
            {
                dealii::Point<3> center((i + 0.5) * size_cell, (j + 0.5) * size_cell, (k + 0.5) * size_cell);
                // set_cube (triangulation, center, size);
                set_ball (triangulation, center, size_inclusion);
            };
        };
    };
};

dbl uber_function (const dealii::Point<2> p, cst n)
{
    cdbl PI = 3.14159265359;
    dbl Uz = 0.0;
    dbl Uw = 0.0;
    for (st i = 1; i < n+1; ++i)
    {
        Uw += (0.25 * 4.0 / (std::pow(PI, 3.0) * std::pow((2.0 * i - 1.0), 3.0)) *
                cosh((2.0 * i - 1.0) * PI * p(1)) / sinh((2.0 * i - 1.0) * PI / 2.0) +
        8.0 / (std::pow(PI, 4.0) * std::pow((2.0 * i - 1.0), 4.0))) *
        cos((2.0 * i - 1.0) * PI * p(0));
    };
    Uz = Uw - 0.25 * (std::pow(p(0) - 0.5, 3.0) / 6.0 - std::pow(p(1), 2.0) / 2.0 * (p(0) - 0.5));
    // printf("Uber %ld %f %f\n", n, Uw, Uz);
    return Uz;
};

int main()
{
    enum {x, y, z};
    debputs();
    lmbd<st(cst)> add_i = [](cst i){return i-1;};
    printf("%ld\n", foo(10, [](cst i){return i-1;}, [](cst i){return i+3;}));
   //  st aa = 1;
   //  st bb = ++aa;
   //  printf("%ld %ld %ld %ld\n", 2_pow(0), aa++, ++aa, bb);
   //  {
   //  CubeOfNumbers<2, 2> cube(5);
   //  printf("%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n", 
   //          cube.number[0][4][0], cube.number[1][4][0], cube.number[2][4][0], cube.number[3][4][0], cube.number[4][4][0],
   //          cube.number[0][3][0], cube.number[1][3][0], cube.number[2][3][0], cube.number[3][3][0], cube.number[4][3][0],
   //          cube.number[0][2][0], cube.number[1][2][0], cube.number[2][2][0], cube.number[3][2][0], cube.number[4][2][0],
   //          cube.number[0][1][0], cube.number[1][1][0], cube.number[2][1][0], cube.number[3][1][0], cube.number[4][1][0],
   //          cube.number[0][0][0], cube.number[1][0][0], cube.number[2][0][0], cube.number[3][0][0], cube.number[4][0][0]);
   //  };
   //  printf("\n");
   //  {
   //  CubeOfNumbers<3, 3> cube(1);
   //  // printf("%ld %ld\n%ld %ld\n\n%ld %ld\n%ld %ld\n", 
   //  //         cube.number[0][1][1][0], cube.number[1][1][1][0],
   //  //         cube.number[0][0][1][0], cube.number[1][0][1][0],
   //  //         cube.number[0][1][0][0], cube.number[1][1][0][0],
   //  //         cube.number[0][0][0][0], cube.number[1][0][0][0]);
   // printf("%ld %ld %ld\n%ld %ld %ld\n%ld %ld %ld\n \n%ld %ld %ld\n%ld %ld %ld\n%ld %ld %ld\n \n%ld %ld %ld\n%ld %ld %ld\n%ld %ld %ld\n\n ", 
   //         cube.number[0][0][0][0], cube.number[1][0][0][0], cube.number[2][0][0][0],
   //         cube.number[0][1][0][0], cube.number[1][1][0][0], cube.number[2][1][0][0],
   //         cube.number[0][2][0][0], cube.number[1][2][0][0], cube.number[2][2][0][0],
   //
   //         cube.number[0][0][1][0], cube.number[1][0][1][0], cube.number[2][0][1][0],
   //         cube.number[0][1][1][0], cube.number[1][1][1][0], cube.number[2][1][1][0],
   //         cube.number[0][2][1][0], cube.number[1][2][1][0], cube.number[2][2][1][0],
   //
   //         cube.number[0][0][2][0], cube.number[1][0][2][0], cube.number[2][0][2][0],
   //         cube.number[0][1][2][0], cube.number[1][1][2][0], cube.number[2][1][2][0],
   //         cube.number[0][2][2][0], cube.number[1][2][2][0], cube.number[2][2][2][0]);
   //  //printf("%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n", 
   //    //      cube.number[0][4][0], cube.number[1][4][0], cube.number[2][4][0], cube.number[3][4][0], cube.number[4][4][0],
   //      //    cube.number[0][3][0], cube.number[1][3][0], cube.number[2][3][0], cube.number[3][3][0], cube.number[4][3][0],
   //        //  cube.number[0][2][0], cube.number[1][2][0], cube.number[2][2][0], cube.number[3][2][0], cube.number[4][2][0],
   //        //  cube.number[0][1][0], cube.number[1][1][0], cube.number[2][1][0], cube.number[3][1][0], cube.number[4][1][0],
   //        //  cube.number[0][0][0], cube.number[1][0][0], cube.number[2][0][0], cube.number[3][0][0], cube.number[4][0][0]);
   //  };
   //  debputs();
    {
        Domain<3> domain;
        {
            dealii::GridGenerator::hyper_cube(domain.grid, 0.0, 2.0);
            domain.grid.refine_global(1);
        };
    printf("level %d\n", domain.grid.n_global_levels());
    };

    // {
    //     dealii::Triangulation<2> tria;
    //     vec<prmt::Point<2>> outer(4);//3,0000000000000004440892098500626161694527
    //     vec<prmt::Point<2>> inner(4);

    //     outer[0].x() = 0.0; outer[0].y() = 0.0;
    //     outer[1].x() = 3.0; outer[1].y() = 0.0;
    //     outer[2].x() = 3.0; outer[2].y() = 3.0;
    //     outer[3].x() = 0.0; outer[3].y() = 3.0;

    //     inner[0].x() = 2.9; inner[0].y() = 2.7;
    //     inner[1].x() = 3.0; inner[1].y() = 2.74;
    //     inner[2].x() = 3.0000000000000003; inner[2].y() = 3.0;
    //     inner[3].x() = 2.77; inner[3].y() = 3.0;
    //     set_grid (tria, outer, inner)0;
    //     {
    //         std::ofstream out ("grid-igor.eps");
    //         dealii::GridOut grid_out;
    //         grid_out.write_eps (tria, out);
    //     };
    // };

    // printf("TRUE? %d\n", 9.0000000000000006 == 9.0000000000000008);
    // puts("TRUE? 0.000000000000000");
    // printf("TRUE? %.50f\n", 3.0);
    // printf("TRUE? %.50f\n", 3.000000000000007);
    // printf("TRUE? %.50f\n", 3.0000000000000007);
    // printf("TRUE? %.50f\n", 3.0000000000000007);
    // printf("TRUE? %.50f\n", (3.0000000000000007 - 3.0));

    //HEAT_CONDUCTION_PROBLEM
    if (false)
    {
        Domain<2> domain;
        {
            dealii::GridGenerator::hyper_cube(domain.grid);
            // vec<prmt::Point<2>> boundary_of_segments;
            // vec<st> types_boundary_segments;
            // arr<st, 4> types_boundary = {0, 0, 0, 0};
            // cst num_segments = 2;
            // prmt::Point<2> p1(0.0, 0.0);
            // prmt::Point<2> p2(1.0, 1.0);
            // debputs();
            // GTools::give_rectangle_with_border_condition (
            //         boundary_of_segments, types_boundary_segments, 
            //         types_boundary, num_segments, p1, p2);
            // debputs();
            // make_grid (domain.grid, boundary_of_segments, types_boundary_segments);
            domain.grid.refine_global(3);
        };
        debputs();
        dealii::FE_Q<2> fe(1);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
        {
            element_matrix.C .resize(1);
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
        };

        auto func = [] (dealii::Point<2>) {return -2.0;};
        SourceScalar<2> element_rhsv (func, domain.dof_handler.get_fe());

        // Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
        // Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);

        // vec<BoundaryValueScalar<2>> bound (1);
        // bound[0].function      = [] (const dealii::Point<2> &p) {return p(0) * p(0);};
        // bound[0].boundary_id   = 0;
        // bound[0].boundary_type = TBV::Dirichlet;

        // for (auto b : bound)
        //     ATools ::apply_boundary_value_scalar<2> (b) .to_slae (slae, domain);

        // dealii::SolverControl solver_control (10000, 1e-12);
        // dealii::SolverCG<> solver (solver_control);
        // solver.solve (
        //         slae.matrix,
        //         slae.solution,
        //         slae.rhsv
        //         ,dealii::PreconditionIdentity()
        //         );

        dealii::Vector<dbl> indexes(slae.rhsv.size());
        {
            cu8 dofs_per_cell = element_rhsv .get_dofs_per_cell ();

            std::vector<u32> local_dof_indices (dofs_per_cell);

            auto cell = domain.dof_handler.begin_active();
            auto endc = domain.dof_handler.end();
            for (; cell != endc; ++cell)
            {
                cell ->get_dof_indices (local_dof_indices);

                FOR (i, 0, dofs_per_cell)
                    indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
            };
        };
        HCPTools ::print_temperature<2> (indexes, domain.dof_handler, "temperature.gpd");
        // HCPTools ::print_temperature<2> (slae.solution, domain.dof_handler, "temperaturei.gpd");
        // HCPTools ::print_heat_conductions<2> (
        //         slae.solution, element_matrix.C, domain, "heat_conductions");
        // HCPTools ::print_heat_gradient<2> (
        //         slae.solution, element_matrix.C, domain, "heat_gradient");
    };


    //HEAT_CONDUCTION_PROBLEM_ON_CELL
    if (false)
    {
        FILE *F;
        F = fopen("square.gpd", "w");
        dbl size = 0.05;
        // while (size*size < 0.6)
        // for (st i = 0; i < 7; ++i)
        {
            Domain<2> domain;
            {
                // vec<prmt::Point<2>> outer(4);
                // vec<prmt::Point<2>> inner(4);

                // outer[0].x() = 0.0; outer[0].y() = 0.0;
                // outer[1].x() = 1.0; outer[1].y() = 0.0;
                // outer[2].x() = 1.0; outer[2].y() = 1.0;
                // outer[3].x() = 0.0; outer[3].y() = 1.0;

                // inner[0].x() = 0.25; inner[0].y() = 0.25;
                // inner[1].x() = 0.75; inner[1].y() = 0.25;
                // inner[2].x() = 0.75; inner[2].y() = 0.75;
                // inner[3].x() = 0.25; inner[3].y() = 0.75;

                // // inner[0].x() = 0.0; inner[0].y() = 0.0;
                // // inner[1].x() = 0.5; inner[1].y() = 0.0;
                // // inner[2].x() = 0.5; inner[2].y() = 1.0;
                // // inner[3].x() = 0.0; inner[3].y() = 1.0;

                // set_grid (domain.grid, outer, inner);

                const size_t material_id[4][4] =
                {
                    {0, 0, 0, 0},
                    {0, 1, 1, 0},
                    {0, 1, 1, 0},
                    {0, 0, 0, 0}
                };
                const double dot[5] = 
                {
                    (0.0),
                    (0.5 - size / 2.0),
                    (0.5),
                    (0.5 + size / 2.0),
                    (1.0)
                };
                // ::set_tria <5> (domain.grid, dot, material_id);
                set_circ(domain.grid, 0.2, 5); //0.344827, 2);
                // set_circ_in_hex(domain.grid, 0.3, 6);
                // ::set_hexagon_grid_pure (domain.grid, 1.0, 0.5);
                // domain.grid .refine_global (3);
                {
                    std::ofstream out ("grid-igor.eps");
                    dealii::GridOut grid_out;
                    grid_out.write_eps (domain.grid, out);
                };
            };
            dealii::FE_Q<2> fe(1);
            domain.dof_init (fe);

            OnCell::SystemsLinearAlgebraicEquations<2> slae;
            OnCell::BlackOnWhiteSubstituter bows;
            // BlackOnWhiteSubstituter bows;

            LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
            // {
            element_matrix.C .resize(2);
            element_matrix.C[1][x][x] = 0.1;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 0.1;
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
            // };
            const bool scalar_type = 0;
            // OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);
            OnCell::prepare_system_equations_with_cubic_grid<2, 1> (slae, bows, domain);
            {
                OnCell::BlackOnWhiteSubstituter bows_old;
                OnCell::BlackOnWhiteSubstituter bows_new;

                {
                    dealii::CompressedSparsityPattern c_sparsity (
                            domain.dof_handler.n_dofs());

                    dealii::DoFTools ::make_sparsity_pattern (
                            domain.dof_handler, c_sparsity);

                    ::OnCell::DomainLooper<2, 0> dl;
                    dl .loop_domain(
                            domain.dof_handler,
                            bows_old,
                            c_sparsity);
                };

                {
                    dealii::CompressedSparsityPattern c_sparsity (
                            domain.dof_handler.n_dofs());

                    dealii::DoFTools ::make_sparsity_pattern (
                            domain.dof_handler, c_sparsity);

                    ::OnCell::DomainLooperTrivial<2, 1> dl;
                    dl .loop_domain(
                            domain.dof_handler,
                            bows_new,
                            c_sparsity);
                };
                printf("Size %ld %ld\n", bows_old.size, bows_new.size);
                for (st i = 0; i < bows_old.size; ++i)
                {
                    printf("%ld %ld %ld  %ld %ld %ld\n", 
                            bows_old.black[i], 
                            bows_new.black[i],
                            bows_new.black[i] - bows_old.black[i],
                            bows_old.white[i], 
                            bows_new.white[i],
                            bows_new.white[i] - bows_old.white[i]
                            );
                };
            };

            OnCell::Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler, bows);
            // FILE *F;
            // F = fopen("matrix.gpd", "w");
            // for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
            //     for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
            //         if (slae.matrix.el(i,j))
            //         {
            //             fprintf(F, "%ld %ld %f\n", i, j, slae.matrix(i,j));
            //         };
            // fclose(F);

            FOR(i, 0, 2)
            {
                vec<arr<dbl, 2>> coef_for_rhs(2);
                FOR(j, 0, element_matrix.C.size())
                {
                    FOR(k, 0, 2)
                    {
                        coef_for_rhs[j][k] = element_matrix.C[j][i][k];
                    };
                };
                OnCell::SourceScalar<2> element_rhsv (coef_for_rhs, domain.dof_handler.get_fe());
                // Assembler::assemble_rhsv<2> (slae.rhsv[i], element_rhsv, domain.dof_handler);
                OnCell::Assembler::assemble_rhsv<2> (slae.rhsv[i], element_rhsv, domain.dof_handler, bows);
                // for (auto a : slae.rhsv[i])
                //     printf("%f\n", a);
                {
                    dealii::DataOut<2> data_out;
                    data_out.attach_dof_handler (domain.dof_handler);
                    data_out.add_data_vector (slae.rhsv[0], "xb");
                    data_out.add_data_vector (slae.rhsv[1], "yb");
                    data_out.build_patches ();

                    auto name = "b.gpd";

                    std::ofstream output (name);
                    data_out.write_gnuplot (output);
                };

                dealii::SolverControl solver_control (500000, 1e-12);
                dealii::SolverCG<> solver (solver_control);
                solver.solve (
                        slae.matrix,
                        slae.solution[i],
                        slae.rhsv[i]
                        ,dealii::PreconditionIdentity()
                        );
                FOR(j, 0, slae.solution[i].size())
                    slae.solution[i][j] = slae.solution[i][bows.subst (j)];
            };
            // {
            //     FILE* F;
            //     F = fopen("Sol.gpd", "w");
            //     for (size_t i = 0; i < slae.solution[0].size(); ++i)
            //         fprintf(F, "%ld %.10f\n", i, slae.solution[0](i));
            //     fclose(F);
            // };

            arr<str, 2> vr = {"temperature_x.gpd", "temperature_y.gpd"};
            FOR(i, 0, 2)
                HCPTools ::print_temperature<2> (slae.solution[i], domain.dof_handler, vr[i]);

            auto meta_coef = OnCell::calculate_meta_coefficients_scalar<2> (
                    domain.dof_handler, slae.solution, slae.rhsv, element_matrix.C);
            printf("META %.15f %.15f %.15f\n", meta_coef[x][x], meta_coef[y][y], meta_coef[x][y]);
            // fprintf(F, "%f %f %f\n", size*size, meta_coef[x][x], meta_coef[y][y]);
            // puts("111111111");
            // size+=0.05;
            // puts("2222222");
            // printf("%f %f\n", size, size*size);
        };
            fclose(F);
    };

    //ELASSTIC_PROBLEM
    if (false)
    {
        Domain<2> domain;
        {
            // // vec<prmt::Point<2>> boundary_of_segments;
            // // vec<st> types_boundary_segments;
            // // arr<st, 4> types_boundary = {0, 2, 2, 2};
            // // cst num_segments = 1;
            // // prmt::Point<2> p1(0.0, 0.0);
            // // prmt::Point<2> p2(1.0, 1.0);
            // // GTools::give_rectangle_with_border_condition (
            // //         boundary_of_segments, types_boundary_segments, 
            // //         types_boundary, num_segments, p1, p2);
            // // for (st i = 0; i < types_boundary_segments.size(); ++i)
            // // {
            // //     printf("OOOO (%f, %f) (%f, %f) %ld\n", 
            // //             boundary_of_segments[i].x(),
            // //             boundary_of_segments[i].y(),
            // //             boundary_of_segments[i+1].x(),
            // //             boundary_of_segments[i+1].y(),
            // //             types_boundary_segments[i]);
            // // };
            // // make_grid (domain.grid, boundary_of_segments, types_boundary_segments);
            //
            // std::vector< dealii::Point< 2 > > v (4);
            //
            // v[0] = dealii::Point<2>(0.0, 0.0);
            // v[1] = dealii::Point<2>(1.0, 0.0);
            // v[2] = dealii::Point<2>(1.0, 1.0);
            // v[3] = dealii::Point<2>(0.0, 1.0);
            //
            // std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());
            //
            // c[0].vertices[0] = 0; 
            // c[0].vertices[1] = 1; 
            // c[0].vertices[2] = 2;
            // c[0].vertices[3] = 3;
            // c[0].material_id = 0; 
            //
            // dealii::SubCellData b;
            //
            // {
            //     dealii::CellData<1> cell;
            //     cell.vertices[0] = 0;
            //     cell.vertices[1] = 1;
            //     cell.boundary_id = 0;
            //     b.boundary_lines .push_back (cell);
            // };
            // {
            //     dealii::CellData<1> cell;
            //     cell.vertices[0] = 1;
            //     cell.vertices[1] = 2;
            //     cell.boundary_id = 2;
            //     b.boundary_lines .push_back (cell);
            // };
            // {
            //     dealii::CellData<1> cell;
            //     cell.vertices[0] = 2;
            //     cell.vertices[1] = 3;
            //     cell.boundary_id = 1;
            //     b.boundary_lines .push_back (cell);
            // };
            // {
            //     dealii::CellData<1> cell;
            //     cell.vertices[0] = 3;
            //     cell.vertices[1] = 0;
            //     cell.boundary_id = 2;
            //     b.boundary_lines .push_back (cell);
            // };
            // // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 0});
            // // b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 2});
            // // b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
            // // b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 2});
            //
            // dealii::GridReordering<2> ::reorder_cells (c);
            // domain.grid .create_triangulation_compatibility (v, c, b);
            //
            // // domain.grid.refine_global(2);
            dealii::GridGenerator::hyper_cube(domain.grid);
            domain.grid.refine_global(2);
        };
        dealii::FESystem<2,2> fe 
            (dealii::FE_Q<2,2>(1), 2);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (1);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);

        const dbl abld = 
            element_matrix.C[0][x][x][x][x] +
            // element_matrix.C[0][x][x][x][y] +
            element_matrix.C[0][y][x][x][x];
            // element_matrix.C[0][y][x][x][y];
        printf("AAAAAA %f\n", abld);
        arr<std::function<dbl (const dealii::Point<2>&)>, 2> func {
        // [=] (const dealii::Point<2>) {return -2.0*abld;},
        [] (const dealii::Point<2>) {return 0.0;},
        [] (const dealii::Point<2>) {return 0.0;}
        };
        // auto func = [] (dealii::Point<2>) {return arr<dbl, 2>{-2.0, 0.0};};
        SourceVector<2> element_rhsv (func, domain.dof_handler.get_fe());

        Assembler ::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
        Assembler ::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);

        vec<BoundaryValueVector<2>> bound (1);
        bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{p(0), 0.0};};
        bound[0].boundary_id   = 0;
        bound[0].boundary_type = TBV::Dirichlet;
        // bound[0].boundary_type = TBV::Neumann;
        // bound[1].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[1].boundary_id   = 1;
        // bound[1].boundary_type = TBV::Dirichlet;
        // // bound[1].boundary_type = TBV::Neumann;
        // bound[2].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[2].boundary_id   = 2;
        // bound[1].boundary_type = TBV::Dirichlet;
        // // bound[2].boundary_type = TBV::Neumann;
    // printf("CCCC %f\n",element_matrix.C[0][1][1][1][1]);
    // print_tensor<6*6>(element_matrix.C[0]);

        for (auto b : bound)
            ATools ::apply_boundary_value_vector<2> (b) .to_slae (slae, domain);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );
        //
        // dealii::Vector<dbl> indexes(slae.solution.size());
        // {
        //     cu8 dofs_per_cell = element_rhsv .get_dofs_per_cell ();
        //
        //     std::vector<u32> local_dof_indices (dofs_per_cell);
        //
        //     auto cell = domain.dof_handler.begin_active();
        //     auto endc = domain.dof_handler.end();
        //     for (; cell != endc; ++cell)
        //     {
        //         cell ->get_dof_indices (local_dof_indices);
        //         printf("%d \n", dofs_per_cell); 
        //         printf("%d \n", indexes.size());
        //
        //         FOR (i, 0, 4)
        //         {
        //             // printf("%d\n", local_dof_indices[i]);
        //             indexes(local_dof_indices[2 * i]) = cell ->vertex_dof_index (i, 0);
        //             indexes(local_dof_indices[2 * i + 1]) = cell ->vertex_dof_index (i, 1);
        //         };
        //     };
        // };
        // EPTools ::print_move<2> (indexes, domain.dof_handler, "move.gpd");
        EPTools ::print_move<2> (slae.solution, domain.dof_handler, "move.gpd");
    };

    // ELASSTIC_PROBLEM_ON_CELL
    if (false)
    {
        // FILE* F;
        // F = fopen("isotropic_test.gpd", "w");
        // F = fopen("isotropic/circ.gpd", "a");
        // F = fopen("isotropic/cube.gpd", "a");
        // F = fopen("ter.gpd", "a");
        // dbl size = 0.01;
        // while (size < 0.99)
        // // while (size < 0.5)
        // {
            Domain<2> domain;
            {
                // const size_t material_id[4][4] =
                // {
                //     {0, 0, 0, 0},
                //     {0, 1, 1, 0},
                //     {0, 1, 1, 0},
                //     {0, 0, 0, 0}
                // };
                // const double dot[5] = 
                // {
                //     (0.0),
                //     (0.5 - size / 2.0),
                //     (0.5),
                //     (0.5 + size / 2.0),
                //     (1.0)
                // };
                // ::set_tria <5> (domain.grid, dot, material_id);
                // ::set_hexagon_grid_pure (domain.grid, 1.0, size);
                set_circ(domain.grid, 0.475, 4);
                // domain.grid .refine_global (3);
                {
                    std::ofstream out ("grid-igor.eps");
                    dealii::GridOut grid_out;
                    grid_out.write_eps (domain.grid, out);
                };
            };
            dealii::FESystem<2,2> fe (dealii::FE_Q<2,2>(1), 2);
            domain.dof_init (fe);

            OnCell::SystemsLinearAlgebraicEquations<4> slae;
            OnCell::BlackOnWhiteSubstituter bows;

            LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
            element_matrix.C .resize (2);
            EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.2}(element_matrix.C[0]);
            EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.28}(element_matrix.C[1]);

            u8 dim = 2;

            const bool vector_type = 1;
            OnCell::prepare_system_equations<vector_type> (slae, bows, domain);

            OnCell::Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler, bows);

            arr<u8, 4> theta  = {x, y, z, x};
            arr<u8, 4> lambda = {x, y, z, y};

#pragma omp parallel for
            for (st n = 0; n < 4; ++n)
            {
                vec<arr<arr<dbl, 2>, 2>> coef_for_rhs(2);

                for (auto i : {x, y})
                    for (auto j : {x, y})
                        for(st k = 0; k < element_matrix.C.size(); ++k)
                        {
                            coef_for_rhs[k][i][j] = 
                                element_matrix.C[k][i][j][theta[n]][lambda[n]];
                        };

                slae.solution[n] = 0;
                slae.rhsv[n] = 0;

                OnCell::SourceVector<2> element_rhsv (
                        coef_for_rhs, domain.dof_handler.get_fe());
                OnCell::Assembler::assemble_rhsv<2> (
                        slae.rhsv[n], element_rhsv, domain.dof_handler, bows);

                dealii::SolverControl solver_control (10000, 1e-12);
                dealii::SolverCG<> solver (solver_control);
                solver.solve (
                        slae.matrix,
                        slae.solution[n],
                        slae.rhsv[n]
                        ,dealii::PreconditionIdentity()
                        );
                FOR(i, 0, slae.solution[n].size())
                    slae.solution[n][i] = slae.solution[n][bows.subst (i)];
            };

            OnCell::SystemsLinearAlgebraicEquations<2> problem_of_torsion_rod_slae;
            vec<ATools::SecondOrderTensor> coef_for_potr(2);
            for (st i = 0; i < 2; ++i)
            {
                coef_for_potr[i][x][x] = element_matrix.C[i][x][z][x][z];
                coef_for_potr[i][y][y] = element_matrix.C[i][y][z][y][z];
                coef_for_potr[i][x][y] = element_matrix.C[i][x][z][y][z];
                coef_for_potr[i][y][x] = element_matrix.C[i][x][z][y][z];
            };
            solved_heat_problem_on_cell<2> (
                    domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);

            arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
            for (st i = 0; i < 4; ++i)
            {
                EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
            };

            auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
                    domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);

            for (size_t i = 0; i < 9; ++i)
            {
                uint8_t im = i / (dim + 1);
                uint8_t in = i % (dim + 1);

                for (size_t j = 0; j < 9; ++j)
                {
                    uint8_t jm = j / (dim + 1);
                    uint8_t jn = j % (dim + 1);

                    if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
                        printf("\x1B[31m%f\x1B[0m   ", 
                                meta_coef[im][in][jm][jn]);
                    else
                        printf("%f   ", 
                                meta_coef[im][in][jm][jn]);
                };
                for (size_t i = 0; i < 2; ++i)
                    printf("\n");
            };
            // print_tensor<6*6>(meta_coef);
            {
            auto newcoef = unphysical_to_physicaly (meta_coef);
        // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
        printf("%f %f %f %f %f %f %f %f %f %f %f\n",
                    newcoef[0][0][0][0],
                    newcoef[0][0][1][1],
                    newcoef[0][0][2][2],
                    newcoef[1][1][0][0],
                    newcoef[1][1][1][1],
                    newcoef[1][1][2][2],
                    newcoef[2][2][0][0],
                    newcoef[2][2][1][1],
                    newcoef[2][2][2][2],
                    meta_coef[0][1][0][1],
                    meta_coef[0][2][0][2]
                    );
            };
                    // fprintf(F, "%f %f %f %f\n", 0.0,
                    //         meta_coef[0][0][0][0],
                    //         meta_coef[0][0][1][1],
                    //         meta_coef[0][1][0][1]);
            // {

            //     st n = 200;
            //     cdbl pi = 3.14159265358979323846;
            //     cdbl d_phi = (1 * pi) / n;
            //     FOR(e, 0, n+1)
            //     {
            //         cdbl phi = e*d_phi;
            //         dbl U[3][3] = {
            //             {cos(phi), -sin(phi), 0.0},
            //             {sin(phi),  cos(phi), 0.0},
            //             {0.0,            0.0, 1.0}};

            //         ATools::FourthOrderTensor pivot_coef;
            //         for (st i = 0; i < 3; ++i)
            //         for (st j = 0; j < 3; ++j)
            //         for (st k = 0; k < 3; ++k)
            //         for (st l = 0; l < 3; ++l)
            //             pivot_coef[i][j][k][l] = 0.0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //         // FOR(i, 0, 3) FOR(j, 0, 3) FOR(k, 0, 3) FOR(l, 0, 3)
            //         //     pivot_coef[i][j][k][l] = SUM(a,0,3,SUM(b,0,3,SUM(c,0,3,SUM(d,0,3,
            //         //                         U[i][a]*U[j][b]*U[k][c]*U[k][d]*meta_coef[a][b][c][d]))));
            //         for (st i = 0; i < 3; ++i)
            //         for (st j = 0; j < 3; ++j)
            //         for (st k = 0; k < 3; ++k)
            //         for (st l = 0; l < 3; ++l)
            //         for (st a = 0; a < 3; ++a)
            //         for (st b = 0; b < 3; ++b)
            //         for (st c = 0; c < 3; ++c)
            //         for (st d = 0; d < 3; ++d)
            //             pivot_coef[i][j][k][l] += 
            //                                 U[i][a]*U[j][b]*U[l][c]*U[k][d]*meta_coef[a][b][c][d];


            //         fprintf(F, "%f %f %f %f %f %f\n", phi,
            //                 pivot_coef[0][0][0][0],
            //                 pivot_coef[0][0][1][1],
            //                 pivot_coef[0][1][0][1],
            //             std::abs((pivot_coef[0][0][0][0] - pivot_coef[0][0][1][1])/2.0 - pivot_coef[0][1][0][1])/pivot_coef[0][1][0][1],
            //             std::abs((pivot_coef[0][0][0][0] - pivot_coef[0][0][1][1])/2.0 - pivot_coef[0][1][0][1]));

            //         // auto newcoef = unphysical_to_physicaly (pivot_coef);
            //         // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", phi, 
            //         //         newcoef[0][0][0][0],
            //         //         newcoef[0][0][1][1],
            //         //         newcoef[0][0][2][2],
            //         //         newcoef[1][1][0][0],
            //         //         newcoef[1][1][1][1],
            //         //         newcoef[1][1][2][2],
            //         //         newcoef[2][2][0][0],
            //         //         newcoef[2][2][1][1],
            //         //         newcoef[2][2][2][2],
            //         //         meta_coef[0][1][0][1],
            //         //         meta_coef[0][2][0][2]
            //         //        );
            //     };
            // };
        //     {
        //         cdbl pi = 3.14159265359;
        //         dbl Y1 = 0.0;
        //         dbl Y2 = 0.0;
        //
        //         {
        //             auto newcoef = unphysical_to_physicaly (meta_coef);
        //             Y1 = newcoef[0][0][0][0];
        //         };
        //
        //         {
        //             cdbl phi = pi / 4.0;
        //             dbl U[3][3] = {
        //                 {cos(phi), -sin(phi), 0.0},
        //                 {sin(phi),  cos(phi), 0.0},
        //                 {0.0,            0.0, 1.0}};
        //
        //             ATools::FourthOrderTensor pivot_coef;
        //
        //             for (st i = 0; i < 3; ++i)
        //             for (st j = 0; j < 3; ++j)
        //             for (st k = 0; k < 3; ++k)
        //             for (st l = 0; l < 3; ++l)
        //                 pivot_coef[i][j][k][l] = 0.0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        //             FOR(i, 0, 3) FOR(j, 0, 3) FOR(k, 0, 3) FOR(l, 0, 3)
        //                 pivot_coef[i][j][k][l] = SUM(a,0,3,SUM(b,0,3,SUM(c,0,3,SUM(d,0,3,
        //                                     U[i][a]*U[j][b]*U[l][c]*U[k][d]*meta_coef[a][b][c][d]))));
        //
        //             auto newcoef = unphysical_to_physicaly (pivot_coef);
        //             Y2 = newcoef[0][0][0][0];
        //         };
        //         fprintf(F, "%f %f %f %f\n", 
        //                 // size*size*pi,
        //                 size*size,
        //                 (Y1 - Y2) / Y1, 
        //                 std::abs((meta_coef[0][0][0][0] - meta_coef[0][0][1][1])/2.0 - meta_coef[0][1][0][1])/meta_coef[0][1][0][1],
        //                 std::abs((meta_coef[1][1][1][1] - meta_coef[1][1][0][0])/2.0 - meta_coef[0][1][0][1])/meta_coef[0][1][0][1]);
        //         // fprintf(F, "%f %f\n", size*size, (Y1 - Y2) / Y1);
        //     };
        //     size += 0.01;
        //     // size += 0.005;
        // };
        // fclose(F);
    };
    
    //HEAT_CONDUCTION_NIKOLA_PROBLEM
    if (1)
    {
        Domain<2> domain;
        // {
        //     vec<prmt::Point<2>> boundary_of_segments;
        //     vec<st> types_boundary_segments;
        //     arr<st, 4> types_boundary = {0, 1, 2, 3}; //clockwise
        //     cst num_segments = 1;
        //     prmt::Point<2> p1(0.0, 0.0);
        //     prmt::Point<2> p2(1.0, 1.0);
        //     debputs();
        //     GTools::give_rectangle_with_border_condition (
        //             boundary_of_segments, types_boundary_segments, 
        //             types_boundary, num_segments, p1, p2);
        //     debputs();
        //     make_grid (domain.grid, boundary_of_segments, types_boundary_segments);
        //     domain.grid.refine_global(3);
        //     // for (st i = 0; i < types_boundary_segments.size(); ++i)
        //     // {
        //     //     printf("%ld\n",types_boundary_segments[i]);
        //     // };
        //     // puts("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        // };
        {

            std::vector< dealii::Point< 2 > > v (6);

            v[0]  = dealii::Point<2>(0.0, -0.5);
            v[1]  = dealii::Point<2>(1.0, -0.5);
            v[2]  = dealii::Point<2>(0.0, 0.0);
            v[3]  = dealii::Point<2>(1.0, 0.0);
            v[4]  = dealii::Point<2>(0.0, 0.5);
            v[5]  = dealii::Point<2>(1.0, 0.5);
            // v[0]  = dealii::Point<2>(0.0, 0.0);
            // v[1]  = dealii::Point<2>(0.0, 1.0);
            // v[2]  = dealii::Point<2>(0.5, 0.0);
            // v[3]  = dealii::Point<2>(0.5, 1.0);
            // v[4]  = dealii::Point<2>(1.0, 0.0);
            // v[5]  = dealii::Point<2>(1.0, 1.0);

            std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());
            {
                dealii::CellData<2> tmp;
                tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.vertices[2]=3;tmp.vertices[3]=2;tmp.material_id=0;
                c .push_back (tmp);
                tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.vertices[2]=4;tmp.vertices[3]=2;tmp.material_id=1;
                c .push_back (tmp);
            };
            // c .push_back (dealii::CellData<2>{{0, 1, 3, 2}, 0});
            // c .push_back (dealii::CellData<2>{{3, 5, 4, 2}, 1});

            dealii::SubCellData b;
            {
                dealii::CellData<1> tmp;
                tmp.vertices[0]=4;tmp.vertices[1]=2;tmp.boundary_id=0;
                b.boundary_lines .push_back (tmp);
                tmp.vertices[0]=2;tmp.vertices[1]=0;tmp.boundary_id=0;
                b.boundary_lines .push_back (tmp);
                tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.boundary_id=1;
                b.boundary_lines .push_back (tmp);
                tmp.vertices[0]=1;tmp.vertices[1]=3;tmp.boundary_id=2;
                b.boundary_lines .push_back (tmp);
                tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.boundary_id=2;
                b.boundary_lines .push_back (tmp);
                tmp.vertices[0]=5;tmp.vertices[1]=4;tmp.boundary_id=3;
                b.boundary_lines .push_back (tmp);
            };
            // b.boundary_lines .push_back (dealii::CellData<1>{4, 2, 0});
            // b.boundary_lines .push_back (dealii::CellData<1>{2, 0, 0});
            // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
            // b.boundary_lines .push_back (dealii::CellData<1>{1, 3, 2});
            // b.boundary_lines .push_back (dealii::CellData<1>{3, 5, 2});
            // b.boundary_lines .push_back (dealii::CellData<1>{5, 4, 3});

            dealii::GridReordering<2> ::reorder_cells (c);
            domain.grid .create_triangulation_compatibility (v, c, b);

            domain.grid .refine_global (5);
        }
        debputs();
        dealii::FE_Q<2> fe(1);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
        {
            element_matrix.C .resize(2);
            element_matrix.C[0][x][x] = 0.4;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 0.4;
            element_matrix.C[1][x][x] = 0.4;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 0.4;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
        };

        // T1.1
        // vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
        // U[0][x] = [] (const dealii::Point<2> &p) {return 1.0;}; //Ux
        // U[0][y] = [] (const dealii::Point<2> &p) {return 0.0;}; //Uy
        // U[1][x] = [] (const dealii::Point<2> &p) {return 1.0;};
        // U[1][y] = [] (const dealii::Point<2> &p) {return 0.0;};
        // vec<typename Nikola::SourceScalar<2>::Func> tau(2);
        // tau[0] = [] (const dealii::Point<2> &p) {return 0.0;};
        // tau[1] = [] (const dealii::Point<2> &p) {return 0.0;};

        // T1.2
        cdbl c0 = 0.5;
        cdbl E = 1.0;
        cdbl nu = 0.25;
        cdbl mu = 0.4;
        vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
        U[0][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));}; //Ux
        U[0][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*(p(0)-c0)*p(1);}; //Uy
        U[1][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));};
        U[1][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*(p(0)-c0)*p(1);};
        // U[0][x] = [] (const dealii::Point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;}; //Ux
        // U[0][y] = [] (const dealii::Point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);}; //Uy
        // U[1][x] = [] (const dealii::Point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;};
        // U[1][y] = [] (const dealii::Point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);};
        vec<typename Nikola::SourceScalar<2>::Func> tau(2);
        tau[0] = [E, c0] (const dealii::Point<2> &p) {return E*(p(0)-c0);};
        tau[1] = [E, c0] (const dealii::Point<2> &p) {return E*(p(0)-c0);};
        // tau[0] = [] (const dealii::Point<2> &p) {return 1.0*p(0);};
        // tau[1] = [] (const dealii::Point<2> &p) {return 1.0*p(0);};
        // tau[0] = [] (const dealii::Point<2> &p) {return -1+0.4*0.25*p(0)+0.25*p(0);};
        // tau[1] = [] (const dealii::Point<2> &p) {return -1+0.4*0.25*p(0)+0.25*p(0);};

        // vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
        // U[0][x] = [] (const dealii::Point<2> &p) {return 0.0;}; //Ux
        // U[0][y] = [] (const dealii::Point<2> &p) {return 0.0;}; //Uy
        // U[1][x] = [] (const dealii::Point<2> &p) {return 0.0;};
        // U[1][y] = [] (const dealii::Point<2> &p) {return 0.0;};
        // vec<typename Nikola::SourceScalar<2>::Func> tau(2);
        // tau[0] = [] (const dealii::Point<2> &p) {return -2.0;};
        // tau[1] = [] (const dealii::Point<2> &p) {return -2.0;};
        Nikola::SourceScalar<2> element_rhsv (U, tau, domain.dof_handler.get_fe());
        // auto func = [] (dealii::Point<2>) {return 0.0;};
        // SourceScalar<2> element_rhsv (func, domain.dof_handler.get_fe());

        Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
        Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);

        // puts("AAAAAAAAAAAAAAAAAAAAA");
        // vec<BoundaryValueScalar<2>> bound (4);
        // bound[0].function      = [] (const dealii::Point<2> &p) {return 2.0*p(0);};
        // // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[0].boundary_id   = 0;
        // bound[0].boundary_type = TBV::Neumann;
        // // bound[0].function      = [] (const dealii::Point<2> &p) {return 0.0;};
        // // bound[0].boundary_id   = 0;
        // // bound[0].boundary_type = TBV::Dirichlet;
        // // bound[1].function      = [] (const dealii::Point<2> &p) {return 0.0;};
        // // bound[1].boundary_id   = 1;
        // // bound[1].boundary_type = TBV::Dirichlet;
        // // bound[2].function      = [] (const dealii::Point<2> &p) {return 1.0;};
        // // bound[2].boundary_id   = 2;
        // // bound[2].boundary_type = TBV::Dirichlet;
        // bound[1].function      = [] (const dealii::Point<2> &p) {return 2.0*p(0);};
        // // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{1.0, 0.0};};
        // bound[1].boundary_id   = 1;
        // bound[1].boundary_type = TBV::Neumann;
        // bound[2].function      = [] (const dealii::Point<2> &p) {return 2.0*p(0);};
        // // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[2].boundary_id   = 2;
        // bound[2].boundary_type = TBV::Neumann;
        // bound[3].function      = [] (const dealii::Point<2> &p) {return 2.0*p(0);};
        // // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{-1.0, 0.0};};
        // bound[3].boundary_id   = 3;
        // bound[3].boundary_type = TBV::Neumann;
        // puts("BBBBBBBBBBBBBb");

        // for (auto b : bound)
        //     ATools ::apply_boundary_value_scalar<2> (b) .to_slae (slae, domain);

        HCPTools ::print_temperature<2> (slae.rhsv, domain.dof_handler, "b");
        dbl sum = 0.0;
        for (st i = 0; i < slae.rhsv.size(); ++i)
        {
            sum += slae.rhsv(i);
        };
        printf("Integral %f\n", sum);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );

        dealii::Vector<dbl> uber(slae.solution.size());
        dealii::Vector<dbl> diff(slae.solution.size());
        for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                dbl indx = cell->vertex_dof_index(i, 0);
                uber(indx) = uber_function(cell->vertex(i), 10);
                diff(indx) = std::abs(uber(indx) - slae.solution(indx));
            };
        };

        HCPTools ::print_temperature<2> (slae.solution, domain.dof_handler, "temperature.gpd");
        HCPTools ::print_temperature<2> (uber, domain.dof_handler, "uber.gpd");
        HCPTools ::print_temperature<2> (diff, domain.dof_handler, "uber-diff.gpd");
        // HCPTools ::print_heat_conductions<2> (
        //         slae.solution, element_matrix.C, domain, "heat_conductions");
        // HCPTools ::print_heat_gradient<2> (
        //         slae.solution, element_matrix.C, domain, "heat_gradient");
        // for (st i = 0; i < 4; ++i)
        //     uber_function (dealii::Point<2>(-0.5, -0.5), i);
        // for (st i = 0; i < 4; ++i)
        //     uber_function (dealii::Point<2>(-0.5, 0.5), i);
        // for (st i = 0; i < 40; ++i)
        //     uber_function (dealii::Point<2>(0.0, 0.0), i);
        cdbl d = 
            (uber_function(dealii::Point<2>(1.0+0.001, 0), 1) - 
            uber_function(dealii::Point<2>(1.0-0.001, 0), 1)) / (0.002);
        printf("Border %f %f\n", d, 0.4 * (d + 0.125 * (0.5*0.5 + 0.5 * 0.5)));
    };

    //HEAT_CONDUCTION_PROBLEM_3D
    if (false)
    {
        Domain<3> domain;
        {
            // dealii::GridGenerator::hyper_cube(domain.grid, 0.0, 2.0);
            // domain.grid.refine_global(1);
            set_cylinder(domain.grid, 0.344827, x, 4);
        };
        debputs();
        dealii::FE_Q<3> fe(1);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianScalar<3> element_matrix (domain.dof_handler.get_fe());
        {
            element_matrix.C .resize(2);
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            element_matrix.C[1][x][x] = 10.0;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 10.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
        };

        auto func = [] (dealii::Point<3>) {return 0.0;};
        SourceScalar<3> element_rhsv (func, domain.dof_handler.get_fe());

        Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler);
        Assembler::assemble_rhsv<3> (slae.rhsv, element_rhsv, domain.dof_handler);

        vec<BoundaryValueScalar<3>> bound (1);
        bound[0].function      = [] (const dealii::Point<3> &p) {return p(1);};
        bound[0].boundary_id   = 0;
        bound[0].boundary_type = TBV::Dirichlet;

        for (auto b : bound)
            ATools ::apply_boundary_value_scalar<3> (b) .to_slae (slae, domain);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );

        // dealii::Vector<dbl> indexes(slae.solution.size());
        // {
        //     cu8 dofs_per_cell = element_rhsv .get_dofs_per_cell ();
        //
        //     std::vector<u32> local_dof_indices (dofs_per_cell);
        //
        //     auto cell = domain.dof_handler.begin_active();
        //     auto endc = domain.dof_handler.end();
        //     for (; cell != endc; ++cell)
        //     {
        //         cell ->get_dof_indices (local_dof_indices);
        //
        //         FOR (i, 0, dofs_per_cell)
        //             indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
        //     };
        // };
        HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "temperature.gpd", dealii::DataOutBase::gnuplot);
        HCPTools ::print_temperature_slice (slae.solution, domain.dof_handler, "temperature_slice.gpd", y, 0.5);
        HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "temperature.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_heat_conductions<2> (
        //         slae.solution, element_matrix.C, domain, "heat_conductions");
        // HCPTools ::print_heat_gradient<2> (
        //         slae.solution, element_matrix.C, domain, "heat_gradient");
    };

    //HEAT_CONDUCTION_PROBLEM_ON_CELL_3D
    if (false)
    {
        FILE *F;
        F = fopen("square.gpd", "w");
        dbl size = 0.05;
        {
            Domain<3> domain;
            {
                // set_cylinder(domain.grid, 0.475, z, 5);
                set_ball(domain.grid, 0.4, 5);
            };
            dealii::FE_Q<3> fe(1);
            domain.dof_init (fe);

            OnCell::SystemsLinearAlgebraicEquations<3> slae;
            OnCell::BlackOnWhiteSubstituter bows;
            // BlackOnWhiteSubstituter bows;

            LaplacianScalar<3> element_matrix (domain.dof_handler.get_fe());
            // {
            element_matrix.C .resize(2);
            element_matrix.C[1][x][x] = 0.1;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 0.1;
            element_matrix.C[1][z][z] = 0.1;
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            element_matrix.C[0][z][z] = 1.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
            // };
    debputs();
            const bool scalar_type = 0;
            // OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);
            OnCell::prepare_system_equations_with_cubic_grid<3, 1> (slae, bows, domain);
    debputs();

            OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

            FOR(i, 0, 3)
            {
                vec<arr<dbl, 3>> coef_for_rhs(2);
                FOR(j, 0, element_matrix.C.size())
                {
                    FOR(k, 0, 3)
                    {
                        coef_for_rhs[j][k] = element_matrix.C[j][i][k];
                    };
                };
                OnCell::SourceScalar<3> element_rhsv (coef_for_rhs, domain.dof_handler.get_fe());
                // Assembler::assemble_rhsv<2> (slae.rhsv[i], element_rhsv, domain.dof_handler);
                OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[i], element_rhsv, domain.dof_handler, bows);
                // for (auto a : slae.rhsv[i])
                //     printf("%f\n", a);
                // {
                //     dealii::DataOut<2> data_out;
                //     data_out.attach_dof_handler (domain.dof_handler);
                //     data_out.add_data_vector (slae.rhsv[0], "xb");
                //     data_out.add_data_vector (slae.rhsv[1], "yb");
                //     data_out.build_patches ();
                //
                //     auto name = "b.gpd";
                //
                //     std::ofstream output (name);
                //     data_out.write_gnuplot (output);
                // };

                dealii::SolverControl solver_control (500000, 1e-12);
                dealii::SolverCG<> solver (solver_control);
                solver.solve (
                        slae.matrix,
                        slae.solution[i],
                        slae.rhsv[i]
                        ,dealii::PreconditionIdentity()
                        );
                FOR(j, 0, slae.solution[i].size())
                    slae.solution[i][j] = slae.solution[i][bows.subst (j)];
            };
            debputs();
            // {
            //     FILE* F;
            //     F = fopen("Sol.gpd", "w");
            //     for (size_t i = 0; i < slae.solution[0].size(); ++i)
            //         fprintf(F, "%ld %.10f\n", i, slae.solution[0](i));
            //     fclose(F);
            // };

            {
                arr<str, 3> vr = {"temperature_x.gpd", "temperature_y.gpd", "temperature_z.gpd"};
                FOR(i, 0, 3)
                    HCPTools ::print_temperature<3> (slae.solution[i], domain.dof_handler, vr[i]);
            };
            {
                arr<str, 3> vr = {"temperature_x.vtk", "temperature_y.vtk", "temperature_z.vtk"};
                FOR(i, 0, 3)
                    HCPTools ::print_temperature<3> (slae.solution[i], domain.dof_handler, vr[i], dealii::DataOutBase::vtk);
            };

            auto meta_coef = OnCell::calculate_meta_coefficients_scalar<3> (
                    domain.dof_handler, slae.solution, slae.rhsv, element_matrix.C);
            printf("META %.15f %.15f %.15f\n", meta_coef[x][x], meta_coef[y][y], meta_coef[x][y]);
            printf("META %.15f %.15f %.15f %.15f %.15f %.15f\n", 
                    meta_coef[x][x], meta_coef[y][y], meta_coef[z][z],
                    meta_coef[x][y], meta_coef[x][z], meta_coef[y][z]);
            // fprintf(F, "%f %f %f\n", size*size, meta_coef[x][x], meta_coef[y][y]);
            // puts("111111111");
            // size+=0.05;
            // puts("2222222");
            // printf("%f %f\n", size, size*size);
        };
            fclose(F);
    };

    // ELASSTIC_PROBLEM_3D
    if (0)
    {
        cdbl len_rod = 1.0;
        Domain<3> domain;
        {
            // dealii::GridGenerator::hyper_cube(domain.grid, 0.0, 1.0);
            // domain.grid.refine_global(2);
            // set_cylinder(domain.grid, 0.475, z, 5);
            // set_ball(domain.grid, 0.4, 3);
            // set_long_rod(domain.grid, len_rod, 0.4, 3);
            // domain.grid.refine_global(2);
            set_speciment(domain.grid, len_rod, 1.0, 1.0, 0.4 / 8.0, 1.0 / 8.0, arr<st,6>({2,2,2,2,2,2}), 5);
        };
        dealii::FESystem<3,3> fe 
            (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        EPTools ::set_isotropic_elascity{yung : 100.0, puasson : 0.15}(element_matrix.C[0]);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);

        const dbl abld = 
            element_matrix.C[0][x][x][x][x] +
            // element_matrix.C[0][x][x][x][y] +
            element_matrix.C[0][y][x][x][x];
            // element_matrix.C[0][y][x][x][y];
        printf("AAAAAA %f\n", abld);
        arr<std::function<dbl (const dealii::Point<3>&)>, 3> func {
        // [=] (const dealii::Point<2>) {return -2.0*abld;},
        [] (const dealii::Point<3>) {return 0.0;},
        [] (const dealii::Point<3>) {return 0.0;},
        [] (const dealii::Point<3>) {return 0.0;}
        };
        // auto func = [] (dealii::Point<2>) {return arr<dbl, 2>{-2.0, 0.0};};
        SourceVector<3> element_rhsv (func, domain.dof_handler.get_fe());

        Assembler ::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler);
        Assembler ::assemble_rhsv<3> (slae.rhsv, element_rhsv, domain.dof_handler);

        vec<BoundaryValueVector<3>> bound (2);
        bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{-1.0, 0.0, 0.0};};
        bound[0].boundary_id   = 1;
        // bound[0].boundary_type = TBV::Dirichlet;
        bound[0].boundary_type = TBV::Neumann;
        bound[1].function      = [] (const dealii::Point<3> &p) {
            // return arr<dbl, 3>{0.0, 0.0, 0.0};};
            if (p(0) == 1.0)
                return arr<dbl, 3>{1.0, 0.0, 0.0};
            else if (p(0) == 0.0)
                return arr<dbl, 3>{-1.0, 0.0, 0.0};
            else
                return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[1].boundary_id   = 2;
        // bound[1].boundary_type = TBV::Dirichlet;
        bound[1].boundary_type = TBV::Neumann;

        for (auto b : bound)
            ATools ::apply_boundary_value_vector<3> (b) .to_slae (slae, domain);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );
        // EPTools ::print_move<3> (slae.solution, domain.dof_handler, "move.gpd");
        // HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "move.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_temperature_slice (slae.solution, domain.dof_handler, "temperature_slice.gpd", x, len_rod);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice.gpd", y, 1.0);
        dbl sum_x = 0.0;
        dbl sum_y = 0.0;
        st n_x = 0;
        st n_y = 0;
        for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<3>::vertices_per_cell; ++i)
            {
                if (std::abs(cell->vertex(i)(x) - 1.0) < 1e-10)
                {
                    sum_x += slae.solution(cell->vertex_dof_index(i, x)); 
                    n_x++;
                };
                if (std::abs(cell->vertex(i)(y) - 1.0) < 1e-10)
                {
                    sum_y += slae.solution(cell->vertex_dof_index(i, y)); 
                    n_y++;
                };
            };
        };
        printf("%f %f %d %d %f %f E=%f nu=%f\n", 
                sum_x, sum_y, n_x, n_y, sum_x/n_x*2.0, sum_y/n_y*2.0, 0.5/(sum_x/n_x), (sum_y/n_y)/(sum_x/n_x));
};

    // ELASSTIC_PROBLEM_ON_CELL_3D
    if (0)
    {
        Domain<3> domain;
        {
            // set_cylinder(domain.grid, 0.475, z, 4);
            set_ball(domain.grid, 0.4, 4);
        };
        dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        OnCell::SystemsLinearAlgebraicEquations<6> slae;
        OnCell::BlackOnWhiteSubstituter bows;

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.2}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.28}(element_matrix.C[1]);
        EPTools ::set_isotropic_elascity{yung : 100.0, puasson : 0.25}(element_matrix.C[0]);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);

        u8 dim = 2;

        // const bool vector_type = 1;
        // OnCell::prepare_system_equations<vector_type> (slae, bows, domain);
        OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);

        OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

        arr<u8, 6> theta  = {x, y, z, x, x, y};
        arr<u8, 6> lambda = {x, y, z, y, z, z};

#pragma omp parallel for
        for (st n = 0; n < 6; ++n)
        {
            vec<arr<arr<dbl, 3>, 3>> coef_for_rhs(2);

            for (auto i : {x, y, z})
                for (auto j : {x, y, z})
                    for(st k = 0; k < element_matrix.C.size(); ++k)
                    {
                        coef_for_rhs[k][i][j] = 
                            element_matrix.C[k][i][j][theta[n]][lambda[n]];
                    };

            slae.solution[n] = 0;
            slae.rhsv[n] = 0;

            OnCell::SourceVector<3> element_rhsv (
                    coef_for_rhs, domain.dof_handler.get_fe());
            OnCell::Assembler::assemble_rhsv<3> (
                    slae.rhsv[n], element_rhsv, domain.dof_handler, bows);

            dealii::SolverControl solver_control (10000, 1e-12);
            dealii::SolverCG<> solver (solver_control);
            solver.solve (
                    slae.matrix,
                    slae.solution[n],
                    slae.rhsv[n]
                    ,dealii::PreconditionIdentity()
                    );
            FOR(i, 0, slae.solution[n].size())
                slae.solution[n][i] = slae.solution[n][bows.subst (i)];
        };

        // arr<str, 6> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd", "move_xz", "move_yz"};
        // for (st i = 0; i < 6; ++i)
        // {
        //     EPTools ::print_move<3> (slae.solution[i], domain.dof_handler, vr[i]);
        // };

        auto meta_coef = OnCell::calculate_meta_coefficients_3d_elastic<3> (
                domain.dof_handler, slae, element_matrix.C);

        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (dim + 1);
            uint8_t in = i % (dim + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (dim + 1);
                uint8_t jn = j % (dim + 1);

                if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
                    printf("\x1B[31m%f\x1B[0m   ", 
                            meta_coef[im][in][jm][jn]);
                else
                    printf("%f   ", 
                            meta_coef[im][in][jm][jn]);
            };
            for (size_t i = 0; i < 2; ++i)
                printf("\n");
        };
        // print_tensor<6*6>(meta_coef);
        {
        auto newcoef = unphysical_to_physicaly (meta_coef);
        // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
        printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
                newcoef[0][0][0][0],
                newcoef[0][0][1][1],
                newcoef[0][0][2][2],
                newcoef[1][1][0][0],
                newcoef[1][1][1][1],
                newcoef[1][1][2][2],
                newcoef[2][2][0][0],
                newcoef[2][2][1][1],
                newcoef[2][2][2][2],
                meta_coef[0][1][0][1],
                meta_coef[0][2][0][2]
                );
        };
    };



    // {
    //     vec<ATools::FourthOrderTensor> C(2);
    //     EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.34}(C[0]);
    //     EPTools ::set_isotropic_elascity{yung : 5.0, puasson : 0.30}(C[1]);
    //     
    //     auto arith = [C] (cdbl area) {
    //         ATools::FourthOrderTensor res;
    //         for (st i = 0; i < 3; ++i)
    //         {
    //             for (st j = 0; j < 3; ++j)
    //             {
    //                 for (st k = 0; k < 3; ++k)
    //                 {
    //                     for (st l = 0; l < 3; ++l)
    //                     {
    //                         res[i][j][k][l] = 
    //                             C[0][i][j][k][l] * (1.0 - area) + 
    //                             C[1][i][j][k][l] * area;
    //                     };
    //                 };
    //             };
    //         };
    //         return res;
    //     };

    //     auto harm = [C] (cdbl area) {
    //         ATools::FourthOrderTensor res;
    //         for (st i = 0; i < 3; ++i)
    //         {
    //             for (st j = 0; j < 3; ++j)
    //             {
    //                 for (st k = 0; k < 3; ++k)
    //                 {
    //                     for (st l = 0; l < 3; ++l)
    //                     {
    //                         res[i][j][k][l] = 1.0 /
    //                             (C[0][i][j][k][l] * area + 
    //                              C[1][i][j][k][l] * (1.0 - area));
    //                     };
    //                 };
    //             };
    //         };
    //         return res;
    //     };

    //     FILE *F1;
    //     FILE *F2;
    //     F1 = fopen("arith.gpd", "a");
    //     F2 = fopen("harm.gpd", "a");
    //     dbl size = 0.01;
    //     while (size < 0.999)
    //     {
    //         auto mean1 = arith(size*size);
    //         auto mean2 = harm(size*size);
    //         auto newcoef1 = unphysical_to_physicaly (mean1);
    //         auto newcoef2 = unphysical_to_physicaly (mean2);

    //         fprintf(F1, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
    //                 newcoef1[0][0][0][0],
    //                 newcoef1[0][0][1][1],
    //                 newcoef1[0][0][2][2],
    //                 newcoef1[1][1][0][0],
    //                 newcoef1[1][1][1][1],
    //                 newcoef1[1][1][2][2],
    //                 newcoef1[2][2][0][0],
    //                 newcoef1[2][2][1][1],
    //                 newcoef1[2][2][2][2],
    //                 mean1[0][1][0][1],
    //                 mean1[0][2][0][2],
    //                 );

    //         fprintf(F2, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
    //                 newcoef2[0][0][0][0],
    //                 newcoef2[0][0][1][1],
    //                 newcoef2[0][0][2][2],
    //                 newcoef2[1][1][0][0],
    //                 newcoef2[1][1][1][1],
    //                 newcoef2[1][1][2][2],
    //                 newcoef2[2][2][0][0],
    //                 newcoef2[2][2][1][1],
    //                 newcoef2[2][2][2][2],
    //                 mean2[0][1][0][1],
    //                 mean2[0][2][0][2]
    //                 );

    //         size += 0.01;
    //     };
    //     fclose(F1);
    //     fclose(F2);
    // };

    
    
//     {
//     arr<prmt::Point<2>, 4> points1 = {
//         prmt::Point<2>(0.0, 0.0),
//         prmt::Point<2>(2.0, 0.0),
//         prmt::Point<2>(1.5, 1.5),
//         prmt::Point<2>(0.0, 2.0)};
//     arr<dbl, 4> values1 = {
//         points1[0].x()*points1[0].x(),
//         points1[1].x()*points1[1].x(),
//         points1[2].x()*points1[2].x(),
//         points1[3].x()*points1[3].x()};
// 
//     arr<prmt::Point<2>, 4> points2 = {
//         prmt::Point<2>(2.0, 0.0),
//         prmt::Point<2>(4.0, 3.0),
//         prmt::Point<2>(3.0, 4.0),
//         prmt::Point<2>(1.5, 1.5)};
//     arr<dbl, 4> values2 = {
//         points2[0].x()*points2[0].x(),
//         points2[1].x()*points2[1].x(),
//         points2[2].x()*points2[2].x(),
//         points2[3].x()*points2[3].x()};
//     // printf("%lf %lf %lf %lf\n", 
//     //         values[0],
//     //         values[1],
//     //         values[2],
//     //         values[3]);
// 
//     Scalar4PointsFunc<2> f1(points1, values1);
// 
//     // points[0] = prmt::Point<2>(1.0, 0.0);
//     // points[1] = prmt::Point<2>(2.0, 0.0);
//     // points[2] = prmt::Point<2>(3.0, 1.0);
//     // points[3] = prmt::Point<2>(2.0, 1.0);
// 
//     // values[0] = points[0].x()*points[0].x();
//     // values[1] = points[1].x()*points[1].x();
//     // values[2] = points[2].x()*points[2].x();
//     // values[3] = points[3].x()*points[3].x();
//     // printf("%lf %lf %lf %lf\n", 
//     //         values[0],
//     //         values[1],
//     //         values[2],
//     //         values[3]);
// 
//     Scalar4PointsFunc<2> f2(points2, values2);
// 
//     auto ip = prmt::Point<2>(1.5, 1.5);
// 
//     // printf("%lf\n", f1(0.0, 0.0));
//     // printf("%lf\n", f1(ip));
//     // printf("%lf\n", f1.dx(ip));
//     // printf("%lf\n", f2(ip));
//     // printf("%lf\n", f2.dx(ip));
// 
// 
//     arr<arr<prmt::Point<2>, 4>, 4> octagon;
//     arr<arr<dbl, 4>, 4> values;
//     dbl angle_delta = 3.14159265359 / 4.0;
//     dbl angle = 0.0;
//     dbl R = 1.0; R /= 10.0;
//     dbl sx = 2.0;
//     dbl sy = 0.0;
// 
//     for (auto& piace : octagon)
//     // for (st i = 0; i < 4; ++i)
//     {
//         piace[0].x() = R * cos(angle) + sx;
//         piace[0].y() = R * sin(angle) + sy;
//         angle += angle_delta;
// 
//         piace[1].x() = R * cos(angle) + sx;
//         piace[1].y() = R * sin(angle) + sy;
//         angle += angle_delta;
// 
//         piace[2].x() = R * cos(angle) + sx;
//         piace[2].y() = R * sin(angle) + sy;
// 
//         piace[3].x() = 0.0 + sx;
//         piace[3].y() = 0.0 + sy;
//     };
//     octagon[0][0].x() += R / 1.0;
//     octagon[3][2].x() += R / 1.0;
//     octagon[0][0].y() += R / 2.0;
//     octagon[3][2].y() += R / 2.0;
// 
//     // octagon[3][1].x() -= R / 2.0;
// 
//     // octagon[0][0] = prmt::Point<2>(-2.0*R + sx, -1.0*R + sy);
//     // octagon[0][1] = prmt::Point<2>(-1.0*R + sx, -1.0*R + sy);
//     // octagon[0][2] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
//     // octagon[0][3] = prmt::Point<2>(-2.0*R + sx, 0.0*R + sy);
// 
//     // octagon[1][0] = prmt::Point<2>(-1.0*R + sx, -1.0*R + sy);
//     // octagon[1][1] = prmt::Point<2>(2.0*R + sx, -1.0*R + sy);
//     // octagon[1][2] = prmt::Point<2>(2.0*R + sx, 0.0*R + sy);
//     // octagon[1][3] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
// 
//     // octagon[2][0] = prmt::Point<2>(-2.0*R + sx, 0.0*R + sy);
//     // octagon[2][1] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
//     // octagon[2][2] = prmt::Point<2>(1.0*R + sx, 1.0*R + sy);
//     // octagon[2][3] = prmt::Point<2>(-2.0*R + sx, 1.0*R + sy);
// 
//     // octagon[3][0] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
//     // octagon[3][1] = prmt::Point<2>(2.0*R + sx, 0.0*R + sy);
//     // octagon[3][2] = prmt::Point<2>(2.0*R + sx, 1.0*R + sy);
//     // octagon[3][3] = prmt::Point<2>(1.0*R + sx, 1.0*R + sy);
// 
// 
//     dbl Rx = 1.0; Rx /= 4.0;
//     dbl Ry = 1.0 / 4.0; Ry /= 4.0;
// 
//     octagon[0][0] = prmt::Point<2>(-2.0*Rx + sx, 0.0*Ry + sy);
//     octagon[0][1] = prmt::Point<2>(-1.0*Rx + sx, -1.0*Ry + sy);
//     octagon[0][2] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
//     octagon[0][3] = prmt::Point<2>(-1.0*Rx + sx, 1.0*Ry + sy);
// 
//     octagon[1][0] = prmt::Point<2>(-1.0*Rx + sx, -1.0*Ry + sy);
//     octagon[1][1] = prmt::Point<2>(-0.5*Rx + sx, -2.0*Ry + sy);
//     octagon[1][2] = prmt::Point<2>(1.0*Rx + sx, -1.0*Ry + sy);
//     octagon[1][3] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
// 
//     octagon[2][0] = prmt::Point<2>(1.0*Rx + sx, -1.0*Ry + sy);
//     octagon[2][1] = prmt::Point<2>(2.0*Rx + sx, 0.0*Ry + sy);
//     octagon[2][2] = prmt::Point<2>(1.0*Rx + sx, 1.0*Ry + sy);
//     octagon[2][3] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
// 
//     octagon[3][0] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
//     octagon[3][1] = prmt::Point<2>(1.0*Rx + sx, 1.0*Ry + sy);
//     octagon[3][2] = prmt::Point<2>(-0.5*Rx + sx, 2.0*Ry + sy);
//     octagon[3][3] = prmt::Point<2>(-1.0*Rx + sx, 1.0*Ry + sy);
// 
//     // octagon[2][2].y() -= R / 1.0;
//     // octagon[3][2].x() += R / 1.0;
// 
//     for (st i = 0; i < 4; ++i)
//     {
//         for (st j = 0; j < 4; ++j)
//         {
//             values[i][j] = 
//                 octagon[i][j].x() * octagon[i][j].x() * octagon[i][j].x() + 2.0
//                 // + octagon[i][j].y() * octagon[i][j].y()
//                 ;
//         };
//     };
// 
//     arr<Scalar4PointsFunc<2>, 4> foo = {
//         Scalar4PointsFunc<2>(octagon[0], values[0]),
//         Scalar4PointsFunc<2>(octagon[1], values[1]),
//         Scalar4PointsFunc<2>(octagon[2], values[2]),
//         Scalar4PointsFunc<2>(octagon[3], values[3])
//     };
// 
//     FILE* F;
//     F = fopen("test_iso.gpd", "w");
//     for (st i = 0; i < 4; ++i)
//     {
//         for (st j = 0; j < 4; ++j)
//         {
//             fprintf(F, "%lf %lf %lf %lf %lf\n",
//                     octagon[i][j].x(), octagon[i][j].y(), values[i][j],
//                     foo[i](octagon[i][j]), foo[i].dx(octagon[i][j]));
//         };
//     };
//     fclose(F);
// 
//     // prmt::Point<2> cp(0.0 + sx, 0.0 + sy);
//     prmt::Point<2> cp(-1.0*Rx + sx, -1.0*Ry + sy);
// 
//     printf("%lf %lf\n", 
//             (foo[0].dx(cp) +
//             foo[1].dx(cp) +
//             foo[2].dx(cp) +
//             foo[3].dx(cp)) / 4.0,
//             // sx*2.0
//             3.0*sx*sx
//             );
// 
//     printf("%lf %lf\n", 
//             foo[1](-1.0*Rx + sx, -1.0*Ry + sy),
//             sx*sx*sx+2.0
//             );
// };
// 
// // {
// //     FILE* F;
// //     F = fopen("test_iso_2.gpd", "w");
// //     dbl R = 1.0;
// //     dbl sx = 1.0;
// //     dbl sy = 0.0;
// //     dbl N = 10;
// //     vec<dbl> angls(N);
// //     for (st i = 0; i < N; ++i)
// //     {
// //         angls[i] = (2.0 * 3.14159265359 / N) * i;
// //     };
// //     for (st i = 0; i < N - 3; ++i)
// //     {
// //         for (st j = i + 1; j < N - 2; ++j)
// //         {
// //             for (st k = j + 1; k < N - 1; ++k)
// //             {
// //                 for (st l = k + 1; l < N; ++l)
// //                 {
// //     arr<prmt::Point<2>, 4> p = {
// //         prmt::Point<2>(R * cos(angls[i]) + sx, R * sin(angls[i]) + sy),
// //         prmt::Point<2>(R * cos(angls[j]) + sx, R * sin(angls[j]) + sy),
// //         prmt::Point<2>(R * cos(angls[k]) + sx, R * sin(angls[k]) + sy),
// //         prmt::Point<2>(R * cos(angls[l]) + sx, R * sin(angls[l]) + sy)};
// //     arr<dbl, 4> v = {
// //         p[0].x()*p[0].x(),
// //         p[1].x()*p[1].x(),
// //         p[2].x()*p[2].x(),
// //         p[3].x()*p[3].x()};
// //     Scalar4PointsFunc<2> foo(p, v);
// //     fprintf(F, "%ld%ld%ld%ld %f %f %f %f %f %f %f %f %f %f %f %f\n",
// //             i, j, k, l, 
// //             p[0].x()*p[0].x(),
// //             p[1].x()*p[1].x(),
// //             p[2].x()*p[2].x(),
// //             p[3].x()*p[3].x(),
// //             foo(p[0]),
// //             foo(p[1]),
// //             foo(p[2]),
// //             foo(p[3]),
// //             (foo(p[0]) - p[0].x()*p[0].x()),
// //             (foo(p[1]) - p[1].x()*p[1].x()),
// //             (foo(p[2]) - p[2].x()*p[2].x()),
// //             (foo(p[3]) - p[3].x()*p[3].x()));
// //                     
// //                 };
// //             };
// //         };
// //     };
// //     fclose(F);
// // };
// {
//     arr<prmt::Point<2>, 4> p = {
//         // prmt::Point<2>(0.0, 0.1),
//         // prmt::Point<2>(0.0125, 0.0875),
//         // prmt::Point<2>(0.040104, 0.096892),
//         // prmt::Point<2>(0.031250, 0.109109)
//         prmt::Point<2>(0.0, 0.0),
//         prmt::Point<2>(2.0, 0.0),
//         prmt::Point<2>(2.0, 2.0),
//         prmt::Point<2>(1.0, 2.0)
//     };
//     arr<dbl, 4> v = {
//         p[0].x() * p[0].x(),
//         p[1].x() * p[1].x(),
//         p[2].x() * p[2].x(),
//         p[3].x() * p[3].x()
//         // 0.0, 0.000161, 0.000912, 0.001574
//     };
//     Scalar4PointsFunc<2> foo(p, v);
//     Beeline foo1(p, v);
//     printf("\n");
//     // printf("AAA f=%f v=%f dif=%f s=%f r=%f Cs=%f Cr=%f\n", 
//     //         foo(p[2]), 
//     //         v[2], 
//     //         foo(p[2]) - v[2], 
//     //         foo.s(p[0].x(), p[0].y()),
//     //         foo.r(p[0].x(), p[0].y()),
//     //         foo.s.C(p[0].x(), p[0].y()),
//     //         foo.r.C(p[0].x(), p[0].y()));
//     cst n = 3;
//     // printf("AAA f=%f v=%f dif=%f s=%f r=%f %f\n", 
//     //         foo(p[n]), 
//     //         v[n], 
//     //         foo(p[n]) - v[n], 
//     //         foo.s(p[n].x(), p[n].y())[0],
//     //         foo.r(p[n].x(), p[n].y())[1], std::sqrt(-4.0));
//     // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
//     //         foo.s(p[0].x(), p[0].y())[0],
//     //         foo.r(p[0].x(), p[0].y())[1],
//     //         foo.s(p[0].x(), p[0].y())[1],
//     //         foo.r(p[0].x(), p[0].y())[0],
//     //         foo.s(p[1].x(), p[1].y())[0],
//     //         foo.r(p[1].x(), p[1].y())[1],
//     //         foo.s(p[1].x(), p[1].y())[1],
//     //         foo.r(p[1].x(), p[1].y())[0],
//     //         foo.s(p[2].x(), p[2].y())[0],
//     //         foo.r(p[2].x(), p[2].y())[1],
//     //         foo.s(p[2].x(), p[2].y())[1],
//     //         foo.r(p[2].x(), p[2].y())[0],
//     //         foo.s(p[3].x(), p[3].y())[0],
//     //         foo.r(p[3].x(), p[3].y())[1],
//     //         foo.s(p[3].x(), p[3].y())[1],
//     //         foo.r(p[3].x(), p[3].y())[0]
//     //         );
//     // dbl midl_x = (0.0 + 0.0125 + 0.040104 + 0.031250) / 4.0;
//     // dbl midl_y = (0.1 + 0.0875 + 0.096892 + 0.109109) / 4.0;
//     // Scalar4PointsFunc<2>::LinFunc(1.0, 2.0, 3.0);
//     // puts("!!!!!!!!");
//     // printf("%f\n", foo.r(p[0].x(), p[0].y())[1]);
//     FILE* F;
//     F = fopen("test_iso_dy.gpd", "w");
//     st NN = 20;
//     dbl summ = 0.0;
//     for (st i = 0; i < NN+1; ++i)
//     {
//         for (st j = 0; j < NN+1; ++j)
//         {
//             dbl x = (p[1].x() - p[0].x()) / NN * i + p[0].x();
//             dbl y = (p[3].y() - p[0].y()) / NN * j + p[0].y();
//             fprintf(F, "%f %f %f %f %f %f %f %f\n", 
//                     x,
//                     y,
//                     foo(x, y),
//                     foo.dy(x, y),
//                     foo1(x, y),
//                     foo1.dy(x, y),
//                     x*x,
//                     2*x);
//             summ += foo.dy(x, y);
//         };
//     };
//     printf("summ %f\n", summ / ((NN+1)*(NN+1)));
//     fclose(F);
//     // dbl midl_x = (p[0].x() + p[1].x() + p[2].x() + p[3].x()) / 4.0;  
//     // dbl midl_y = (p[0].y() + p[1].y() + p[2].y() + p[3].y()) / 4.0;  
//     // printf("%f %f %f %f %f\n", midl_x, midl_y, foo.dy(midl_x, midl_y), foo.dy(0.0, 0.0),
//     //         foo(0.0, 0.05));
//     // puts("!!!!!!!!");
//     // for (st i = 0; i < 11; ++i)
//     // {
//     //     dbl x = 0.0;
//     //     dbl y = -0.02 + i*0.01;
//     //     dbl s = foo.s(x,y)[0];
//     //     dbl r = foo.r(x,y)[1];
//     //     dbl f = foo(x,y);
//     //     printf("%f %f %f %f\n", y, s, r, f);
//     //     // foo(x, y);
//     // };
//     // // foo(0.0, 0.0)
//     // // foo(0.0, 0.05);
// };
// {
//     arr<prmt::Point<2>, 4> p = {
//         prmt::Point<2>(0.0, 0.0),
//         prmt::Point<2>(2.0, 0.0),
//         prmt::Point<2>(2.0, 2.0),
//         prmt::Point<2>(1.0, 2.0)
//     };
//     arr<dbl, 4> v = {
//         p[0].x() * p[0].x(),
//         p[1].x() * p[1].x(),
//         p[2].x() * p[2].x(),
//         p[3].x() * p[3].x()
//     };
//     Scalar4PointsFunc<2> foo1(p, v);
//     Beeline foo2(p, v);
//     printf("%f %f\n", foo1.dy(1.0, 1.0), foo2.dy(1.0, 1.0));
// };

    return EXIT_SUCCESS;
}
