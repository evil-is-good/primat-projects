#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
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
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations_alternate/prepare_system_equations_alternate.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations_with_cubic_grid/prepare_system_equations_with_cubic_grid.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/system_linear_algebraic_equations/system_linear_algebraic_equations.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/calculate_meta_coefficients/calculate_meta_coefficients.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/stress_calculator/stress_calculator.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/deform_calculator/deform_calculator.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/assembler/assembler.h"
// #include "../../../calculation_core/src/blocks/special/problem_on_cell/domain_looper_trivial/domain_looper_trivial.h"

#include "../../../calculation_core/src/blocks/general/laplacian/vector/laplacian_vector.h"
#include "../../../calculation_core/src/blocks/general/source/vector/source_vector.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/apply_boundary_value/vector/apply_boundary_value_vector.h"
#include "../../../calculation_core/src/blocks/special/elastic_problem_tools/elastic_problem_tools.h"

#include "../../../calculation_core/src/blocks/special/problem_on_cell/source/vector/source_vector.h"


#include "../../../calculation_core/src/blocks/special/nikola_problem/source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/nikola_problem/source/vector/source_vector.h"

#include "../../../calculation_core/src/blocks/special/feature_source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/feature_source/vector/source_vector.h"

#include "../../../calculation_core/src/blocks/special/poly_materials_source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/poly_materials_source/vector/source_vector.h"

#include "../../../calculation_core/src/blocks/general/4_points_function/4_points_function.h"

#include "../../../calculation_core/src/blocks/general/gaus_solver/gaus_solver.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/base/geometry_info.h>

extern void make_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<st>);

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<prmt::Point<2>>);

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<prmt::Point<2>>,
        vec<st>);

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<vec<prmt::Point<2>>>,
        vec<st>);

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

bool point_in_quadrilateral (const dealii::Point<2> &p, const arr<dealii::Point<2>, 4> &quad)
{
    bool res = false;

    auto above_the_line = 
        [p] (const dealii::Point<2, double> pl, 
                const dealii::Point<2, double> pr) -> bool
        {
            const uint8_t x = 0;
            const uint8_t y = 1;

            if ((
                        pl(x) * pr(y) - 
                        pr(x) * pl(y) + 
                        p(x)  * (pl(y) - pr(y)) + 
                        p(y)  * (pr(x) - pl(x))) >= -1e-12)
                return true;
            else
                return false;
        };

    size_t sum_positive_sign = 
        above_the_line(quad[0], quad[1]) +
        above_the_line(quad[1], quad[2]) +
        above_the_line(quad[2], quad[3]) +
        above_the_line(quad[3], quad[0]);

    if (sum_positive_sign == 4)
        res = true;

    return res;
};

bool point_in_cell (const dealii::Point<2> &p, 
        const typename dealii::DoFHandler<2>::active_cell_iterator &cell)
{
    arr<dealii::Point<2>, 4> quad;
    quad[0] = dealii::Point<2>(cell->vertex(0)(0), cell->vertex(0)(1));
    quad[1] = dealii::Point<2>(cell->vertex(1)(0), cell->vertex(1)(1));
    quad[2] = dealii::Point<2>(cell->vertex(3)(0), cell->vertex(3)(1));
    quad[3] = dealii::Point<2>(cell->vertex(2)(0), cell->vertex(2)(1));

    return point_in_quadrilateral (p, quad);
};

bool point_in_cell (const dealii::Point<2> &p, 
        const typename dealii::Triangulation<2>::active_cell_iterator &cell)
{
    arr<dealii::Point<2>, 4> quad;
    quad[0] = dealii::Point<2>(cell->vertex(0)(0), cell->vertex(0)(1));
    quad[1] = dealii::Point<2>(cell->vertex(1)(0), cell->vertex(1)(1));
    quad[2] = dealii::Point<2>(cell->vertex(3)(0), cell->vertex(3)(1));
    quad[3] = dealii::Point<2>(cell->vertex(2)(0), cell->vertex(2)(1));

    return point_in_quadrilateral (p, quad);
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

    void give_line_without_end_point(
            vec<prmt::Point<2>> &curve,
            cst num_points,
            prmt::Point<2> first,
            prmt::Point<2> second)
    {
        dbl dx = (second.x() - first.x()) / num_points;
        dbl dy = (second.y() - first.y()) / num_points;
        dbl x = first.x();
        dbl y = first.y();
        FOR_I(0, num_points - 0)
        {
            // printf("x=%f y=%f dx=%f dy=%f\n", x, y, dx, dy);
            curve .push_back (prmt::Point<2>(x, y)); 
            x += dx;
            y += dy;
        };
    };

    void give_rectangle(
            vec<prmt::Point<2>> &curve,
            cst num_points_on_edge,
            prmt::Point<2> first,
            prmt::Point<2> second)
    {
        give_line_without_end_point(curve, num_points_on_edge,
                first,
                prmt::Point<2>(first.x(), second.y()));

        give_line_without_end_point(curve, num_points_on_edge,
                prmt::Point<2>(first.x(), second.y()),
                second);

        give_line_without_end_point(curve, num_points_on_edge,
                second,
                prmt::Point<2>(second.x(), first.y()));

        give_line_without_end_point(curve, num_points_on_edge,
                prmt::Point<2>(second.x(), first.y()),
                first);

    };

void give_rectangle_with_border_condition(
        vec<prmt::Point<2>> &curve,
        vec<st> &type_edge,
        const arr<st, 4> type_border,
        cst num_points_on_edge,
        const prmt::Point<2> first,
        const prmt::Point<2> second)
{
    give_line_without_end_point(curve, num_points_on_edge,
            first,
            prmt::Point<2>(first.x(), second.y()));

    give_line_without_end_point(curve, num_points_on_edge,
            prmt::Point<2>(first.x(), second.y()),
            second);

    give_line_without_end_point(curve, num_points_on_edge,
            second,
            prmt::Point<2>(second.x(), first.y()));

    give_line_without_end_point(curve, num_points_on_edge,
            prmt::Point<2>(second.x(), first.y()),
            first);

    cst n_edge_on_border = curve.size() / 4;
    // printf("type %d\n", n_edge_on_border);
    type_edge.resize(curve.size());

    FOR(i, 0, 4)
        FOR(j, 0 + n_edge_on_border * i, n_edge_on_border + n_edge_on_border * i)
        type_edge[j] = type_border[i];
};

    void give_circ(
            vec<prmt::Point<2>> &curve,
            cst num_points_on_tip,
            cdbl radius,
            prmt::Point<2> center)
    {
        cdbl PI = 3.14159265359;
        cdbl angle_step_rad = 2.0 * PI / num_points_on_tip;
        for (
                dbl angle_rad = PI / 2.0; 
                std::abs(angle_rad - 5.0 * (PI / 2.0)) > 1.e-8; 
                angle_rad += angle_step_rad
            )
        {
            dbl X = radius * cos(angle_rad) + center.x();
            dbl Y = radius * sin(angle_rad) + center.y();
            // printf("circ %f %f\n", X, Y);
            curve .push_back (prmt::Point<2>(X, Y)); 
        };

        // for (
        //         dbl angle_rad = 3.0 * (PI / 2.0); 
        //         abs(angle_rad - (2.5 * PI)) > 1.e-8; 
        //         angle_rad += angle_step_rad
        //     )
        // {
        //     dbl X = radius * sin(angle_rad) + center.x();
        //     dbl Y = radius * cos(angle_rad) + center.y();
        //     printf("circ %f %f\n", X, Y);
        //     curve .push_back (prmt::Point<2>(X, Y)); 
        // };

    };

    template <u8 dim>
    void solve_heat_problem_on_cell_aka_torsion_rod (
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

    template <uint8_t dim>
    void set_quadrate (dealii::Triangulation<dim> &triangulation, 
            cdbl x0, cdbl x1, cdbl x2, cdbl x3,
            cdbl y0, cdbl y1, cdbl y2, cdbl y3,
            // const double lower, const double top,
            size_t n_refine)
    {
        // const double x0 = 0.0;
        // const double x1 = lower;
        // const double x2 = top;
        // const double x3 = 128.0;

    //    std::vector< dealii::Point< 2 > > v (8);
    //
    //    v[0][0] = x0; v[0][1] = x0;
    //    v[1][0] = x4; v[1][1] = x0;
    //    v[2][0] = x1; v[2][1] = x1;
    //    v[3][0] = x3; v[3][1] = x1;
    //    v[4][0] = x0; v[4][1] = x4;
    //    v[5][0] = x1; v[5][1] = x3;
    //    v[6][0] = x3; v[6][1] = x3;
    //    v[7][0] = x4; v[7][1] = x4;
    //
    //    std::vector< dealii::CellData< 2 > > c (5, dealii::CellData<2>());
    //
    //    c[0].vertices[0] = 0;
    //    c[0].vertices[1] = 1;
    //    c[0].vertices[2] = 2;
    //    c[0].vertices[3] = 3;
    //    c[0].material_id = 0;
    //
    //    c[1].vertices[0] = 0;
    //    c[1].vertices[1] = 2;
    //    c[1].vertices[2] = 4;
    //    c[1].vertices[3] = 5;
    //    c[1].material_id = 0;
    //    
    //    c[2].vertices[0] = 2;
    //    c[2].vertices[1] = 3;
    //    c[2].vertices[2] = 5;
    //    c[2].vertices[3] = 6;
    //    c[2].material_id = 1;
    //
    //    c[3].vertices[0] = 4;
    //    c[3].vertices[1] = 5;
    //    c[3].vertices[2] = 7;
    //    c[3].vertices[3] = 6;
    //    c[3].material_id = 0;
    //    
    //    c[4].vertices[0] = 1;
    //    c[4].vertices[1] = 7;
    //    c[4].vertices[2] = 3;
    //    c[4].vertices[3] = 6;
    //    c[4].material_id = 0;

        std::vector< dealii::Point< 2 > > v (16);

        v[0]  = dealii::Point<dim>(x0, y0);
        v[1]  = dealii::Point<dim>(x1, y0);
        v[2]  = dealii::Point<dim>(x2, y0);
        v[3]  = dealii::Point<dim>(x3, y0);
        v[4]  = dealii::Point<dim>(x0, y1);
        v[5]  = dealii::Point<dim>(x1, y1);
        v[6]  = dealii::Point<dim>(x2, y1);
        v[7]  = dealii::Point<dim>(x3, y1);
        v[8]  = dealii::Point<dim>(x0, y2);
        v[9]  = dealii::Point<dim>(x1, y2);
        v[10] = dealii::Point<dim>(x2, y2);
        v[11] = dealii::Point<dim>(x3, y2);
        v[12] = dealii::Point<dim>(x0, y3);
        v[13] = dealii::Point<dim>(x1, y3);
        v[14] = dealii::Point<dim>(x2, y3);
        v[15] = dealii::Point<dim>(x3, y3);

        std::vector< dealii::CellData< 2 > > c (9, dealii::CellData<2>());

        c[6].vertices[0] = 8;  c[7].vertices[0] = 9;  c[8].vertices[0] = 10;
        c[6].vertices[1] = 9;  c[7].vertices[1] = 10; c[8].vertices[1] = 11;
        c[6].vertices[2] = 12; c[7].vertices[2] = 13; c[8].vertices[2] = 14;
        c[6].vertices[3] = 13; c[7].vertices[3] = 14; c[8].vertices[3] = 15;
        c[6].material_id = 0;  c[7].material_id = 0;  c[8].material_id = 0;

        c[3].vertices[0] = 4;  c[4].vertices[0] = 5;  c[5].vertices[0] = 6;
        c[3].vertices[1] = 5;  c[4].vertices[1] = 6;  c[5].vertices[1] = 7;
        c[3].vertices[2] = 8;  c[4].vertices[2] = 9;  c[5].vertices[2] = 10;
        c[3].vertices[3] = 9;  c[4].vertices[3] = 10; c[5].vertices[3] = 11;
        c[3].material_id = 0;  c[4].material_id = 1;  c[5].material_id = 0;

        c[0].vertices[0] = 0;  c[1].vertices[0] = 1;  c[2].vertices[0] = 2;
        c[0].vertices[1] = 1;  c[1].vertices[1] = 2;  c[2].vertices[1] = 3;
        c[0].vertices[2] = 4;  c[1].vertices[2] = 5;  c[2].vertices[2] = 6;
        c[0].vertices[3] = 5;  c[1].vertices[3] = 6;  c[2].vertices[3] = 7;
        c[0].material_id = 0;  c[1].material_id = 0;  c[2].material_id = 0;


        triangulation .create_triangulation (v, c, dealii::SubCellData());

        if (n_refine > 0)
            triangulation .refine_global (n_refine);
    };

    ATools::FourthOrderTensor unphysical_to_physicaly (
            ATools::FourthOrderTensor &unphys)
    {
        enum {x, y, z};
        ATools::FourthOrderTensor res;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                for (st k = 0; k < 3; ++k)
                {
                    for (st l = 0; l < 3; ++l)
                    {
                        res[i][j][k][l] = 0.0;
                    };
                };
            };
        };

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

    void set_rect(dealii::Triangulation< 2 > &triangulation, 
            const dealii::Point<2> p1, const dealii::Point<2> p2, const size_t n_refine)
    {
        dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
        // triangulation .refine_global (1);
        // {
        //     for (st i = 0; i < 1; ++i)
        //     {
        //         auto cell = triangulation .begin_active();
        //         auto endc = triangulation .end();
        //         for (; cell != endc; ++cell)
        //         {
        //         cell ->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
        //         };
        //         triangulation.execute_coarsening_and_refinement ();
        //     };
        // };
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

               // printf("%f %f\n", midle_p(0), midle_p(1));

                if 
                    (
                    (
                    (midle_p(0) > p1(0)) and
                    (midle_p(0) < p2(0))
                    ) and
                    (
                    (midle_p(1) > p1(1)) and
                    (midle_p(1) < p2(1))
                    )
                    )
                {
                    cell->set_material_id(1);
                   // puts("adf");
                }
                else
                    cell->set_material_id(0);
            };
        };
    };

    void set_rect_3d(dealii::Triangulation<3> &triangulation, 
            const dealii::Point<2> p1,
            const dealii::Point<2> p2,
            const size_t n_refine)
    {
        dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
        // triangulation .refine_global (1);
        // {
        //     for (st i = 0; i < 1; ++i)
        //     {
        //         auto cell = triangulation .begin_active();
        //         auto endc = triangulation .end();
        //         for (; cell != endc; ++cell)
        //         {
        //         cell ->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
        //         };
        //         triangulation.execute_coarsening_and_refinement ();
        //     };
        // };
        triangulation .refine_global (n_refine);
        {
            dealii::Triangulation<3>::active_cell_iterator
                cell = triangulation .begin_active(),
                     end_cell = triangulation .end();
            for (; cell != end_cell; ++cell)
            {
                dealii::Point<2> midle_p(0.0, 0.0);

                for (size_t i = 0; i < 8; ++i)
                {
                    midle_p(0) += cell->vertex(i)(0);
                    midle_p(1) += cell->vertex(i)(1);
                };
                midle_p(0) /= 8.0;
                midle_p(1) /= 8.0;

               // printf("%f %f\n", midle_p(0), midle_p(1));

                if 
                    (
                    (
                    (midle_p(0) > p1(0)) and
                    (midle_p(0) < p2(0))
                    ) and
                    (
                    (midle_p(1) > p1(1)) and
                    (midle_p(1) < p2(1))
                    )
                    )
                {
                    cell->set_material_id(1);
                   // puts("adf");
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

void set_cylinder_true(dealii::Triangulation< 3 > &triangulation, 
            const double radius, cst ort, cst n_points_on_includ_border, cst n_slices)
{
    dealii::Triangulation<2> tria2d;
    vec<prmt::Point<2>> border;
    vec<st> type_border;
    give_rectangle_with_border_condition(
            border,
            type_border,
            arr<st, 4>{1,3,2,4},
            10,
            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
    vec<vec<prmt::Point<2>>> inclusion(1);
    dealii::Point<2> center (0.5, 0.5);
    give_circ(inclusion[0], n_points_on_includ_border, radius, prmt::Point<2>(center));
    ::set_grid(tria2d, border, inclusion, type_border);

    {
        std::ofstream out ("grid-igor.eps");
        dealii::GridOut grid_out;
        grid_out.write_eps (tria2d, out);
    };

    dealii::GridGenerator::extrude_triangulation (tria2d, n_slices, 1.0, triangulation);

    {
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

template <st size>
arr<arr<dbl,size>,size> inverse (const arr<arr<dbl,size>,size> &in) 
{
    arr<arr<dbl,size>,size> A;
    arr<arr<dbl,size>,size> B;
    for (st i = 0; i < size; ++i)
    {
        for (st j = 0; j < size; ++j)
        {
            A[i][j] = in[i][j];
            if (i == j)
                B[i][i] = 1.0;
            else
                B[i][j] = 0.0;
        };
    };
    // arr<arr<dbl,3>,3> B = {arr<dbl,3>{1.0, 0.0, 0.0},
    //                         arr<dbl,3>{0.0, 1.0, 0.0},
    //                         arr<dbl,3>{0.0, 0.0, 1.0}};
    // for (st i = 0; i < 3; ++i)
    // {
    //     dbl tmp = A[i][i]; 
    //     for (st j = 0; j < 3; ++j)
    //     {
    //         A[i][j] /= tmp;
    //         B[i][j] /= tmp;
    //     };
    // };
    //

    for (st i = 0; i < size-1; ++i)
    {
        {
        dbl tmp = A[i][i]; 
        for (st j = 0; j < size; ++j)
        {
            A[i][j] /= tmp;
            B[i][j] /= tmp;
        };
        };
        for (st j = i+1; j < size; ++j)
        {
            cdbl tmp = A[j][i];
            for (st k = 0; k < size; ++k)
            {
                A[j][k] -= A[i][k] * tmp;
                B[j][k] -= B[i][k] * tmp;
            };
        };
    };
        {
        dbl tmp = A[size-1][size-1]; 
        for (st j = 0; j < size; ++j)
        {
            A[size-1][j] /= tmp;
            B[size-1][j] /= tmp;
        };
        };

    for (st i = size-1; i > 0; --i)
    {
        for (i32 j = i-1; j > -1; --j)
        {
            // printf("%ld %ld\n", i, j);
            cdbl tmp = A[j][i];
            for (st k = 0; k < size; ++k)
            {
                A[j][k] -= A[i][k] * tmp;
                B[j][k] -= B[i][k] * tmp;
            };
        };
    };
    //
    // printf("%f %f %f\n", A[0][0], A[0][1], A[0][2]);
    // printf("%f %f %f\n", A[1][0], A[1][1], A[1][2]);
    // printf("%f %f %f\n", A[2][0], A[2][1], A[2][2]);
    //
    // printf("\n");
    //
    // printf("%f %f %f\n", B[0][0], B[0][1], B[0][2]);
    // printf("%f %f %f\n", B[1][0], B[1][1], B[1][2]);
    // printf("%f %f %f\n", B[2][0], B[2][1], B[2][2]);
     return B;
};

arr<arr<dbl, 6>, 6> t4_to_t2(const ATools::FourthOrderTensor &in)
{
    arr<arr<dbl, 6>, 6> out;
    
    const arr<arr<st, 2>, 6> r = {
        arr<st, 2>{0, 0},
        arr<st, 2>{1, 1},
        arr<st, 2>{2, 2},
        arr<st, 2>{1, 2},
        arr<st, 2>{2, 0},
        arr<st, 2>{0, 1}};

    for (st i = 0; i < 6; ++i)
    {
        for (st j = 0; j < 6; ++j)
        {
            out[i][j] = in[r[i][0]][r[i][1]][r[j][0]][r[j][1]];
        };
    };

    return out;
};

ATools::FourthOrderTensor t2_to_t4(const arr<arr<dbl, 6>, 6> &in)
{
    ATools::FourthOrderTensor out;
    
    const arr<arr<st, 2>, 6> r = {
        arr<st, 2>{0, 0},
        arr<st, 2>{1, 1},
        arr<st, 2>{2, 2},
        arr<st, 2>{1, 2},
        arr<st, 2>{2, 0},
        arr<st, 2>{0, 1}};

    for (st i = 0; i < 6; ++i)
    {
        for (st j = 0; j < 6; ++j)
        {
            out[r[i][0]][r[i][1]][r[j][0]][r[j][1]] = in[i][j];
            out[r[i][0]][r[i][1]][r[j][1]][r[j][0]] = in[i][j];
            out[r[i][1]][r[i][0]][r[j][0]][r[j][1]] = in[i][j];
            out[r[i][1]][r[i][0]][r[j][1]][r[j][0]] = in[i][j];
        };
    };

    return out;
};

//     void set_hole_3d(dealii::Triangulation< 3 > &triangulation, 
//             const double radius, cst ort, const size_t n_refine)
// {
//         std::vector< dealii::Point< 2 > > v (16);
//
//         v[0]  = dealii::Point<dim>(0, 0);
//         v[1]  = dealii::Point<dim>(x1, 0);
//         v[2]  = dealii::Point<dim>(x2, 0);
//         v[3]  = dealii::Point<dim>(x3, 0);
//         v[4]  = dealii::Point<dim>(x0, y1);
//         v[5]  = dealii::Point<dim>(x1, y1);
//         v[6]  = dealii::Point<dim>(x2, y1);
//         v[7]  = dealii::Point<dim>(x3, y1);
//         v[8]  = dealii::Point<dim>(x0, y2);
//         v[9]  = dealii::Point<dim>(x1, y2);
//         v[10] = dealii::Point<dim>(x2, y2);
//         v[11] = dealii::Point<dim>(x3, y2);
//         v[12] = dealii::Point<dim>(x0, y3);
//         v[13] = dealii::Point<dim>(x1, y3);
//         v[14] = dealii::Point<dim>(x2, y3);
//         v[15] = dealii::Point<dim>(x3, y3);
//
//         std::vector< dealii::CellData< 2 > > c (9, dealii::CellData<2>());
//
//         c[6].vertices[0] = 8;  c[7].vertices[0] = 9;  c[8].vertices[0] = 10;
//         c[6].vertices[1] = 9;  c[7].vertices[1] = 10; c[8].vertices[1] = 11;
//         c[6].vertices[2] = 12; c[7].vertices[2] = 13; c[8].vertices[2] = 14;
//         c[6].vertices[3] = 13; c[7].vertices[3] = 14; c[8].vertices[3] = 15;
//         c[6].material_id = 0;  c[7].material_id = 0;  c[8].material_id = 0;
//
//         c[3].vertices[0] = 4;  c[4].vertices[0] = 5;  c[5].vertices[0] = 6;
//         c[3].vertices[1] = 5;  c[4].vertices[1] = 6;  c[5].vertices[1] = 7;
//         c[3].vertices[2] = 8;  c[4].vertices[2] = 9;  c[5].vertices[2] = 10;
//         c[3].vertices[3] = 9;  c[4].vertices[3] = 10; c[5].vertices[3] = 11;
//         c[3].material_id = 0;  c[4].material_id = 1;  c[5].material_id = 0;
//
//         c[0].vertices[0] = 0;  c[1].vertices[0] = 1;  c[2].vertices[0] = 2;
//         c[0].vertices[1] = 1;  c[1].vertices[1] = 2;  c[2].vertices[1] = 3;
//         c[0].vertices[2] = 4;  c[1].vertices[2] = 5;  c[2].vertices[2] = 6;
//         c[0].vertices[3] = 5;  c[1].vertices[3] = 6;  c[2].vertices[3] = 7;
//         c[0].material_id = 0;  c[1].material_id = 0;  c[2].material_id = 0;
//
//
//         triangulation .create_triangulation (v, c, dealii::SubCellData());
//
//         if (n_refine > 0)
//             triangulation .refine_global (n_refine);
// };
//
    void set_ball(dealii::Triangulation< 3 > &triangulation, 
            const double radius, const size_t n_refine)
    {
        dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
       // puts("1111111111111111111111111111111111111111111");
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

    void set_tube(dealii::Triangulation< 2 > &triangulation, const str file_name, 
            const dealii::Point<2> center,
            cdbl inner_radius, cdbl outer_radius, cst n_refine)
    {
        // dealii::GridGenerator ::hyper_ball (triangulation, center, outer_radius);
        dealii::GridIn<2> gridin;
        gridin.attach_triangulation(triangulation);
        std::ifstream f(file_name);
        gridin.read_msh(f);
        // triangulation .refine_global (n_refine);
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

               // printf("%f %f\n", midle_p(0), midle_p(1));

                if (center.distance(midle_p) < inner_radius)
                {
                    cell->set_material_id(1);
    //                puts("adf");
                }
                else
                    cell->set_material_id(0);
            };
        };
    };

    void set_crazy_tube(dealii::Triangulation< 2 > &triangulation, const dealii::Point<2> center,
            cdbl inner_radius, cdbl outer_radius, cst n_refine)
    {
        dealii::GridGenerator ::hyper_cube (triangulation, -2.0, 2.0);
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

               // printf("%f %f\n", midle_p(0), midle_p(1));

                if (center.distance(midle_p) < inner_radius)
                {
                    cell->set_material_id(1);
    //                puts("adf");
                }
                else if (center.distance(midle_p) < outer_radius)
                {
                    cell->set_material_id(0);
    //                puts("adf");
                }
                else
                    cell->set_material_id(2);
            };
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

    template<uint8_t dim>
    dbl get_value (
            const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
            const dealii::Vector<double> solution,
            const dealii::Point<dim, double> &p)
    {
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

        double a=-(x2*y2*x3*y4*f1+y1*x3*x4*y4*f2-y1*x3*x2*y2*f4-x1*y1*x3*y4*f2-x1*y3*x4*y4*f2-x1*f3*x2*y2*y4+x1*x3*y3*f2*y4+x1*y3*x2*y2*f4+y1*f3*x4*x2*y2-y3*x4*x2*y2*f1-y3*y1*x3*x4*f2+x1*y3*x4*y1*f2-x2*y3*x3*y4*f1+x1*y1*f3*y4*x2+y1*y3*x3*x2*f4+x2*y3*x4*y4*f1-y1*f3*y4*x4*x2+x3*y3*x4*y2*f1-x3*f1*y2*x4*y4+y2*x3*x1*y1*f4+y2*x1*f3*y4*x4-y2*x4*x1*y1*f3-y2*x1*y3*x3*f4-x1*y1*y3*x2*f4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

        return  a + b * p(0) + c * p(1) + d * p(0) * p(1);
    };

    dbl get_value_in_domain (
            const dealii::Point<2> &p,
            const typename dealii::DoFHandler<2> &dof_h,
            const dealii::Vector<double> &solution)
    {
        dbl res = 0.0;
        auto cell = dof_h.begin_active();
        auto endc = dof_h.end();
        for (; cell != endc; ++cell)
        {
            if (point_in_cell (p, cell))
            {
                res = get_value<2> (cell, solution, p);
                break;
            };
        };
        return res;
    };

    // template <st dim>
    // class CellHeatProblem
    // {
    //     public:
    //         CellHeatProblem (
    //                 const vec<arr<arr<dbl, dim>, dim>> &coefficient, 
    //                 const Domain<2> &domain;
    //                 );
    //         void calculate_approximate (const arr<st, 3> approximation);
    //
    //         dealii::FE_Q<2> fe(1);
    //         OnCell::SystemsLinearAlgebraicEquations<2> slae;
    //         OnCell::BlackOnWhiteSubstituter bows;
    //
    //         LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
    //         OnCell::prepare_system_equations_with_cubic_grid<2, 1> (slae, bows, domain);
    //
    //         arr<arr<arr<dbl, dim>, dim>, dim> meta_coefficient;
    //         OnCell::CellFunctionScalar psi_func;
    // ;
    // template <st dim>
    //     CellHeatProblem<dim>::CellHeatProblem (
    //                 const vec<arr<arr<dbl, dim>, dim>> &coefficient, 
    //                 const Domain<2> &domain;
    //                 ) :
    //         fe(1),
    //         element_matrix (domain.dof_handler.get_fe())
    //     {
    //     };
    //
    //
    // void get_approx_cell_heat (
    //         const arr<st, 3> approximation,
    //         const vec<arr<arr<dbl, dim>, dim>> &coefficient, 
    //         const Domain<2> &domain;
    //         arr<arr<arr<dbl, dim>, dim>, dim> *meta_coefficient,
    //         CellFunctionScalar *psi_func)
    // {
    //     dealii::FE_Q<2> fe(1);
    //     domain.dof_init (fe);
    //
    //     OnCell::SystemsLinearAlgebraicEquations<2> slae;
    //     OnCell::BlackOnWhiteSubstituter bows;
    //     // BlackOnWhiteSubstituter bows;
    //
    //     LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
    //     // {
    //     element_matrix.C .resize(2);
    //     element_matrix.C[1][x][x] = lambda;
    //     element_matrix.C[1][x][y] = 0.0;
    //     element_matrix.C[1][y][x] = 0.0;
    //     element_matrix.C[1][y][y] = lambda;
    //     element_matrix.C[0][x][x] = 1.0;
    //     element_matrix.C[0][x][y] = 0.0;
    //     element_matrix.C[0][y][x] = 0.0;
    //     element_matrix.C[0][y][y] = 1.0;
    //     // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
    //     // };
    //     const bool scalar_type = 0;
    //     // OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);
    //     OnCell::prepare_system_equations_with_cubic_grid<2, 1> (slae, bows, domain);
    //     // {
    //     //     OnCell::BlackOnWhiteSubstituter bows_old;
    //     //     OnCell::BlackOnWhiteSubstituter bows_new;
    //     //
    //     //     {
    //     //         dealii::CompressedSparsityPattern c_sparsity (
    //     //                 domain.dof_handler.n_dofs());
    //     //
    //     //         dealii::DoFTools ::make_sparsity_pattern (
    //     //                 domain.dof_handler, c_sparsity);
    //     //
    //     //         ::OnCell::DomainLooper<2, 0> dl;
    //     //         dl .loop_domain(
    //     //                 domain.dof_handler,
    //     //                 bows_old,
    //     //                 c_sparsity);
    //     //     };
    //     //
    //     //     {
    //     //         dealii::CompressedSparsityPattern c_sparsity (
    //     //                 domain.dof_handler.n_dofs());
    //     //
    //     //         dealii::DoFTools ::make_sparsity_pattern (
    //     //                 domain.dof_handler, c_sparsity);
    //     //
    //     //         ::OnCell::DomainLooperTrivial<2, 1> dl;
    //     //         dl .loop_domain(
    //     //                 domain.dof_handler,
    //     //                 bows_new,
    //     //                 c_sparsity);
    //     //     };
    //     //     printf("Size %ld %ld\n", bows_old.size, bows_new.size);
    //     //     for (st i = 0; i < bows_old.size; ++i)
    //     //     {
    //     //         printf("%ld %ld %ld  %ld %ld %ld\n", 
    //     //                 bows_old.black[i], 
    //     //                 bows_new.black[i],
    //     //                 bows_new.black[i] - bows_old.black[i],
    //     //                 bows_old.white[i], 
    //     //                 bows_new.white[i],
    //     //                 bows_new.white[i] - bows_old.white[i]
    //     //                 );
    //     //     };
    //     // };
    //
    //     OnCell::Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler, bows);
    //     // FILE *F;
    //     // F = fopen("matrix.gpd", "w");
    //     // for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
    //     //     for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
    //     //         if (slae.matrix.el(i,j))
    //     //         {
    //     //             fprintf(F, "%ld %ld %f\n", i, j, slae.matrix(i,j));
    //     //         };
    //     // fclose(F);
    //
    //     FOR(i, 0, 2)
    //     {
    //         vec<arr<dbl, 2>> coef_for_rhs(2);
    //         FOR(j, 0, element_matrix.C.size())
    //         {
    //             FOR(k, 0, 2)
    //             {
    //                 coef_for_rhs[j][k] = element_matrix.C[j][i][k];
    //             };
    //         };
    //         OnCell::SourceScalar<2> element_rhsv (coef_for_rhs, domain.dof_handler.get_fe());
    //         // Assembler::assemble_rhsv<2> (slae.rhsv[i], element_rhsv, domain.dof_handler);
    //         OnCell::Assembler::assemble_rhsv<2> (slae.rhsv[i], element_rhsv, domain.dof_handler, bows);
    //         // for (auto a : slae.rhsv[i])
    //         //     printf("%f\n", a);
    //         {
    //             dealii::DataOut<2> data_out;
    //             data_out.attach_dof_handler (domain.dof_handler);
    //             data_out.add_data_vector (slae.rhsv[0], "xb");
    //             data_out.add_data_vector (slae.rhsv[1], "yb");
    //             data_out.build_patches ();
    //
    //             auto name = "b.gpd";
    //
    //             std::ofstream output (name);
    //             data_out.write_gnuplot (output);
    //         };
    //
    //         dealii::SolverControl solver_control (500000, 1e-12);
    //         dealii::SolverCG<> solver (solver_control);
    //         solver.solve (
    //                 slae.matrix,
    //                 slae.solution[i],
    //                 slae.rhsv[i]
    //                 ,dealii::PreconditionIdentity()
    //                 );
    //         FOR(j, 0, slae.solution[i].size())
    //             slae.solution[i][j] = slae.solution[i][bows.subst (j)];
    //     };
    // };
    //
void approx_iteration (st number_of_approx, std::function<void(arr<i32, 3>, cst, cst)> func)
{
    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    {
        for (st i = 0; i < approx_number+1; ++i)
        {
            for (st j = 0; j < approx_number+1; ++j)
            {
                for (st k = 0; k < approx_number+1; ++k)
                {
                    if ((i+j+k) == approx_number)
                    {
                        arr<i32, 3> approximation = {i, j, k};
                        for (st nu = 0; nu < 3; ++nu)
                        {
                            for (st alpha = 0; alpha < 3; ++alpha)
                            {
                                func(approximation, nu, alpha);
                            };
                        };
                    };
                };
            };
        };
    };
};

    void solve_heat_conduction_problem (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
            Domain<2> domain;
            {
                // dealii::GridGenerator::hyper_cube(domain.grid);
                // dealii::GridGenerator::hyper_ball(domain.grid, dealii::Point<2>(0.0,0.0), 2.0);
                // dealii::GridGenerator::hyper_shell(domain.grid, dealii::Point<2>(0.0,0.0), 0.0, 2.0);
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
                // domain.grid.refine_global(3);
                // set_tube(domain.grid, dealii::Point<2>(0.0,0.0), 1.0, 2.0, 2);
                // set_tube(domain.grid, str("circle_R2.msh"), dealii::Point<2>(0.0,0.0), 1.0, 2.0, 1);
                    vec<prmt::Point<2>> border;
                    vec<st> type_border;
                    give_rectangle_with_border_condition(
                            border,
                            type_border,
                            arr<st, 4>{1,3,2,4},
                            11,
                            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(10.0, 10.0));
                    vec<vec<prmt::Point<2>>> inclusion(2);
                    give_rectangle(inclusion[0], 1,
                            prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.45, 0.45));
                    give_rectangle(inclusion[1], 1,
                            prmt::Point<2>(0.65, 0.65), prmt::Point<2>(0.75, 0.75));
// 
                    ::set_grid(domain.grid, border, inclusion, type_border);

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

            auto func = [] (dealii::Point<2>) {return 0.0;};
            SourceScalar<2> element_rhsv (func, domain.dof_handler.get_fe());

            Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
            Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);

            vec<BoundaryValueScalar<2>> bound (4);
            bound[0].function      = [] (const dealii::Point<2> &p) {return 1;};
            bound[0].boundary_id   = 1;
            bound[0].boundary_type = TBV::Dirichlet;
            bound[1].function      = [] (const dealii::Point<2> &p) {return 2;};
            bound[1].boundary_id   = 2;
            bound[1].boundary_type = TBV::Dirichlet;
            bound[2].function      = [] (const dealii::Point<2> &p) {return 3;};
            bound[2].boundary_id   = 3;
            bound[2].boundary_type = TBV::Dirichlet;
            bound[3].function      = [] (const dealii::Point<2> &p) {return 4;};
            bound[3].boundary_id   = 4;
            bound[3].boundary_type = TBV::Dirichlet;

            for (auto b : bound)
                ATools ::apply_boundary_value_scalar<2> (b) .to_slae (slae, domain);

            dealii::SolverControl solver_control (10000, 1e-12);
            dealii::SolverCG<> solver (solver_control);
            solver.solve (
                    slae.matrix,
                    slae.solution,
                    slae.rhsv
                    ,dealii::PreconditionIdentity()
                    );

            // dealii::Vector<dbl> indexes(slae.rhsv.size());
            vec<dealii::Point<2>> coor(slae.rhsv.size());
            {
                cu8 dofs_per_cell = element_rhsv .get_dofs_per_cell ();

                std::vector<u32> local_dof_indices (dofs_per_cell);

                auto cell = domain.dof_handler.begin_active();
                auto endc = domain.dof_handler.end();
                for (; cell != endc; ++cell)
                {
                    cell ->get_dof_indices (local_dof_indices);

                    // FOR (i, 0, dofs_per_cell)
                    //     indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
                    FOR (i, 0, dofs_per_cell)
                    {
                        coor[local_dof_indices[i]] = cell ->vertex (i);
                    };
                };
            };
            // dealii::Vector<dbl> sol_dx(slae.rhsv.size());
            // dealii::Vector<dbl> sol_dy(slae.rhsv.size());
            // sol_dx = 0.0;
            // sol_dy = 0.0;
            // FILE *F;
            // F = fopen("1.tmp", "w");
            // for (st i = 0; i < coor.size(); ++i)
            // {
            //     cdbl cx = coor[i](0);
            //     cdbl cy = coor[i](1);
            //     if (
            //             ((cx > 1.0e-10) and (cx < (1.0 - 1.0e-10))) and
            //             ((cy > 1.0e-10) and (cy < (1.0 - 1.0e-10)))
            //        )
            //     {
            //         cdbl dx = 1.0e-5;
            //         cdbl dy = 1.0e-5;
            //         auto px1 = dealii::Point<2>(cx - dx, cy);
            //         auto px2 = dealii::Point<2>(cx + dx, cy);
            //         auto py1 = dealii::Point<2>(cx, cy - dy);
            //         auto py2 = dealii::Point<2>(cx, cy + dy);
            //         cdbl vx1 = get_value_in_domain (px1, domain.dof_handler, slae.solution);
            //         cdbl vx2 = get_value_in_domain (px2, domain.dof_handler, slae.solution);
            //         cdbl vy1 = get_value_in_domain (py1, domain.dof_handler, slae.solution);
            //         cdbl vy2 = get_value_in_domain (py2, domain.dof_handler, slae.solution);
            //         sol_dx[i] = (vx2 - vx1) / (2.0 * dx);
            //         sol_dy[i] = (vy2 - vy1) / (2.0 * dy);
            //         fprintf(F, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
            //                 cx, cy, vx1, vx2, vy1, vy2,
            //                 (vx2 - vx1) / (2.0 * dx),
            //                 (vy2 - vy1) / (2.0 * dy),
            //                 sol_dx[i], sol_dy[i]);
            //     };
            //     // printf("%d %d\n", i, indexes(i));
            // };
            // fclose(F);
            // auto p = dealii::Point<2>(0.03125, 0.53125);
            // auto p = dealii::Point<2>(0.1, 0.5);
            // auto p = dealii::Point<2>(0.65625, 0.65625);
            // printf("res %f\n", get_value_in_domain (p, domain.dof_handler, slae.solution));
            // cdbl cx = p(0);
            // cdbl cy = p(1);
            //         cdbl dx = 1.0e-10;
            //         cdbl dy = 1.0e-10;
            //         auto px1 = dealii::Point<2>(cx - dx, cy);
            //         auto px2 = dealii::Point<2>(cx + dx, cy);
            //         auto py1 = dealii::Point<2>(cx, cy - dy);
            //         auto py2 = dealii::Point<2>(cx, cy + dy);
            //         cdbl vx1 = get_value_in_domain (px1, domain.dof_handler, slae.solution);
            //         cdbl vx2 = get_value_in_domain (px2, domain.dof_handler, slae.solution);
            //         cdbl vy1 = get_value_in_domain (py1, domain.dof_handler, slae.solution);
            //         cdbl vy2 = get_value_in_domain (py2, domain.dof_handler, slae.solution);
            // printf("%.10f %.10f %.10f %.10f %.10f %.10f\n", vx1, vx2, vy1, vy2, (vx2 - vx1) / (2.0 * dx),
            //                 (vy2 - vy1) / (2.0 * dy));
            // HCPTools ::print_temperature<2> (indexes, domain.dof_handler, "temperature.gpd");
            HCPTools ::print_temperature<2> (slae.solution, domain.dof_handler, "temperature.gpd");
            // HCPTools ::print_temperature<2> (sol_dx, domain.dof_handler, "temperature_dx.gpd");
            // HCPTools ::print_temperature<2> (sol_dy, domain.dof_handler, "temperature_dy.gpd");
            // HCPTools ::print_heat_conductions<2> (
            //         slae.solution, element_matrix.C, domain, "heat_conductions");
            HCPTools ::print_heat_gradient<2> (
                    slae.solution, domain, "heat_gradient.gpd");
        };
    };

    void solve_heat_conduction_problem_on_cell (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
            // FILE *F;
            // F = fopen("square.gpd", "w");
            cdbl c = 0.78;
            cdbl lambda = 2.0e0;
            cdbl R_in = sqrt((c / dealii::numbers::PI));
            dbl size = sqrt(c);
            size = 0.5;
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
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
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
                    // domain.grid .refine_global (1);
                    set_rect(domain.grid,
                            dealii::Point<2>((0.5 - 0.5 / 2.0),(0.5 - 1.5 / 2.0)),
                            dealii::Point<2>((0.5 + 0.5 / 2.0),(0.5 + 1.5 / 2.0)), 4);
                    // set_circ(domain.grid, R_in, 6); //0.344827, 2);
                    // set_circ_in_hex(domain.grid, 0.3, 6);
                    // ::set_hexagon_grid_pure (domain.grid, 1.0, 0.5);
                    // {
                    //     std::ofstream out ("grid-igor.eps");
                    //     dealii::GridOut grid_out;
                    //     grid_out.write_eps (domain.grid, out);
                    // };
                };
                dealii::FE_Q<2> fe(1);
                domain.dof_init (fe);

                OnCell::SystemsLinearAlgebraicEquations<2> slae;
                OnCell::BlackOnWhiteSubstituter bows;
                // BlackOnWhiteSubstituter bows;

                LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
                // {
                element_matrix.C .resize(2);
                element_matrix.C[1][x][x] = lambda;
                element_matrix.C[1][x][y] = 0.0;
                element_matrix.C[1][y][x] = 0.0;
                element_matrix.C[1][y][y] = lambda;
                element_matrix.C[0][x][x] = 1.0;
                element_matrix.C[0][x][y] = 0.0;
                element_matrix.C[0][y][x] = 0.0;
                element_matrix.C[0][y][y] = 1.0;
                // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
                // };
                const bool scalar_type = 0;
                // OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);
                OnCell::prepare_system_equations_with_cubic_grid<2, 1> (slae, bows, domain);
                // {
                //     OnCell::BlackOnWhiteSubstituter bows_old;
                //     OnCell::BlackOnWhiteSubstituter bows_new;
                //
                //     {
                //         dealii::CompressedSparsityPattern c_sparsity (
                //                 domain.dof_handler.n_dofs());
                //
                //         dealii::DoFTools ::make_sparsity_pattern (
                //                 domain.dof_handler, c_sparsity);
                //
                //         ::OnCell::DomainLooper<2, 0> dl;
                //         dl .loop_domain(
                //                 domain.dof_handler,
                //                 bows_old,
                //                 c_sparsity);
                //     };
                //
                //     {
                //         dealii::CompressedSparsityPattern c_sparsity (
                //                 domain.dof_handler.n_dofs());
                //
                //         dealii::DoFTools ::make_sparsity_pattern (
                //                 domain.dof_handler, c_sparsity);
                //
                //         ::OnCell::DomainLooperTrivial<2, 1> dl;
                //         dl .loop_domain(
                //                 domain.dof_handler,
                //                 bows_new,
                //                 c_sparsity);
                //     };
                //     printf("Size %ld %ld\n", bows_old.size, bows_new.size);
                //     for (st i = 0; i < bows_old.size; ++i)
                //     {
                //         printf("%ld %ld %ld  %ld %ld %ld\n", 
                //                 bows_old.black[i], 
                //                 bows_new.black[i],
                //                 bows_new.black[i] - bows_old.black[i],
                //                 bows_old.white[i], 
                //                 bows_new.white[i],
                //                 bows_new.white[i] - bows_old.white[i]
                //                 );
                //     };
                // };

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
                //
                auto meta_coef = OnCell::calculate_meta_coefficients_scalar<2> (
                        domain.dof_handler, slae.solution, slae.rhsv, element_matrix.C);
                printf("META %.15f %.15f %.15f\n", meta_coef[x][x], meta_coef[y][y], meta_coef[x][y]);

            //     cdbl A = R_in;
            //     cdbl R_0 = 0.5;
            //     cdbl lambda_1 = element_matrix.C[0][x][x];
            //     cdbl lambda_2 = element_matrix.C[1][x][x];
            //     // auto D = [lamdda_2, R_in, R_0] (cdbl theta){
            //     //     cdbl delta = [](cdbl t){while (PI / 2.0 > t){t - (PI / 2.0)}; return t;}(theta);
            //     //     cdbl chi = (R_in*R_in)/(R_0*R_0) * cos(delta);
            //     //     return lambda_2 + 1.0 - chi * (lambda_2 - 1.0);
            //     // };
            //     // auto C_11 = 
            //     auto solve_andrianov = [A, R_0, lambda_2](const dealii::Point<2> p){
            //             cdbl r = sqrt(p(x)*p(x) + p(y)*p(y));
            //             cdbl theta = ((std::abs(r) > 1e-5) ? acos(p(x) / r) : 0.0);// - dealii::numbers::PI / 2.0;
            //             cdbl delta = [](dbl t){
            //                 while ((dealii::numbers::PI / 4.0) < t)
            //                 {
            //                     t -= (dealii::numbers::PI / 2.0);
            //                 };
            //                 return std::abs(t);}(theta);
            //             cdbl R_theta = R_0 / cos(delta);
            //             cdbl chi = (A*A) / (R_theta*R_theta);
            //             cdbl D = lambda_2 + 1.0 - chi * (lambda_2 - 1.0);
            //             cdbl C_11 = (lambda_2 - 1.0) * chi / D;
            //             // cdbl C_12 = (lambda_2 - 1.0) * A * A / D;
            //             cdbl C_12 = - (C_11 * A * A) / chi;
            //             cdbl C_21 = - (lambda_2 - 1.0) * (1.0 - chi) / D;
            //             
            //             return (r < A) ? C_21*r : C_11*r+C_12/r;
            //     };
            //
            //     dealii::Vector<dbl> andrianov(slae.solution[0].size());
            //     dealii::Vector<dbl> andrianov_x(slae.solution[0].size());
            //     dealii::Vector<dbl> andrianov_y(slae.solution[0].size());
            //     dealii::Vector<dbl> t_r(slae.solution[0].size());
            //     dealii::Vector<dbl> t_dx(slae.solution[0].size());
            //     dealii::Vector<dbl> diff(slae.solution[0].size());
            //     dealii::Vector<dbl> andrianov_dx(slae.solution[0].size());
            //     for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
            //     {
            //         for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            //         {
            //             cdbl indx_x = cell->vertex_dof_index(i, x);
            //             // dbl indx_y = cell->vertex_dof_index(i, y);
            //             auto p = cell->vertex(i);
            //             p(x) -= 0.5;
            //             p(y) -= 0.5;
            //             
            //             cdbl r = std::pow(p(x)*p(x) + p(y)*p(y), 0.5);
            //             // cdbl theta = atan (p(y) / p(x));
            //             cdbl theta = ((std::abs(r) > 1e-5) ? acos(p(x) / r) : 0.0);// - dealii::numbers::PI / 2.0;
            //             // cdbl cos = ((std::abs(r) > 1e-5) ? (p(x) / r) : 1.0);// - dealii::numbers::PI / 2.0;
            //             // printf("chi %f %f %f\n", cos, p(x), r);
            //             cdbl delta = [](dbl t){
            //                 while ((dealii::numbers::PI / 4.0) < t)
            //                 // while (t > 1.0/sqrt(2.0))
            //                 {
            //                     t -= (dealii::numbers::PI / 2.0);
            //                     // t -= sqrt(2.0);
            //                     // printf("theta %f %f\n", (dealii::numbers::PI / 2.0), t);
            //                 };
            //                 return std::abs(t);}(theta);
            //             cdbl R_theta = R_0 / cos(delta);
            //             // cdbl chi = (A*A)/(R_0*R_0) * 
            //             //     //cos * cos;
            //             //     cos(delta) * cos(delta);
            //             cdbl chi = (A*A) / (R_theta*R_theta);
            //             cdbl D = lambda_2 + 1.0 - chi * (lambda_2 - 1.0);
            //             cdbl C_11 = (lambda_2 - 1.0) * chi / D;
            //             // cdbl C_12 = (lambda_2 - 1.0) * A * A / D;
            //             cdbl C_12 = - (C_11 * A * A) / chi;
            //             cdbl C_21 = - (lambda_2 - 1.0) * (1.0 - chi) / D;
            //             // if (std::abs(1.107149 - delta) < 1e-5)
            //             // printf("theta %f %f delta %f %f chi %f D %f  %f %f  %f r %f R_theta %f\n", 
            //             //         theta, theta / dealii::numbers::PI,
            //             //         delta, delta / dealii::numbers::PI,
            //             //         chi, D, C_11*r, C_12/r,
            //             //         chi * R_theta * R_theta - A * A,
            //             //         r, R_theta);
            //             // if (p(x) == 0.5)
            //             // if (std::abs(r - R_0) < 1e-5)
            //                 // printf("C %f %f %f %f %f %f\n", D, C_11*r, C_12/r, C_11, C_12, chi);
            //
            //             andrianov(indx_x) = (r < A) ? C_21*r : C_11*r+C_12/r;
            //             // andrianov(indx_x) = solve_andrianov(p);
            //             andrianov_x(indx_x) = andrianov(indx_x) * cos(theta);
            //             andrianov_y(indx_x) = std::abs(r) > 1e-5 ?
            //                 andrianov(indx_x) * p(y) / r :
            //                 0.0;
            //             // printf("sin %f %f\n", sin(theta), theta / dealii::numbers::PI);
            //             // andrianov_y(indx_x) = andrianov(indx_x) * cos(dealii::numbers::PI / 2 - theta);
            //             t_r(indx_x) = std::abs(r) > 1e-5 ? 
            //                 // slae.solution(indx_x) * p(0) / r + slae.solution(indx_y) * p(1) / r :
            //                 std::sqrt(std::pow(slae.solution[x](indx_x), 2.0) + std::pow(slae.solution[y](indx_x), 2.0)) :
            //                 0.0;
            //             // if (p(0) == -0.5)
            //             //     printf("%f %f %f %f %f %f\n", digit(indx_x),
            //             //             slae.solution(indx_x),
            //             //             p(0) / r,
            //             //             slae.solution(indx_y),
            //             //             p(1) / r,
            //             //             r);
            //             // digit(indx_y) = C1 * r;
            //             diff(indx_x) = 
            //                 std::abs(andrianov(indx_x) - t_r(indx_x));
            //         };
            //     };
            //     HCPTools ::print_temperature<2> (t_r, domain.dof_handler, "temperature_r.gpd");
            //     HCPTools ::print_temperature<2> (andrianov, domain.dof_handler, "andrianov.gpd");
            //     HCPTools ::print_temperature<2> (andrianov_x, domain.dof_handler, "andrianov_x.gpd");
            //     HCPTools ::print_temperature<2> (andrianov_y, domain.dof_handler, "andrianov_y.gpd");
            //     HCPTools ::print_temperature<2> (diff, domain.dof_handler, "diff.gpd");
            //     for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
            //     {
            //         for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            //         {
            //             cdbl indx = cell->vertex_dof_index(i, x);
            //             cst mat = cell->material_id();
            //             andrianov_dx[indx] = 
            //                 element_matrix.C[mat][x][x] * get_grad<2>(cell, andrianov_x, i)[x];
            //             t_dx[indx] = 
            //                 element_matrix.C[mat][x][x] * get_grad<2>(cell, slae.solution[x], i)[x];
            //         };
            //     };
            //     HCPTools ::print_temperature<2> (andrianov_dx, domain.dof_handler, "andrianov_dx.gpd");
            //     HCPTools ::print_temperature<2> (t_dx, domain.dof_handler, "t_dx.gpd");
            //
            //     dbl sum = 0.0;
            //     for (st i = 0; i < andrianov.size(); ++i)
            //     {
            //         sum += andrianov_dx[i];
            //     };
            //     sum /= andrianov.size();
            //     printf("andrianov meta %f %f\n", sum, sum + meta_coef[y][y]);
            //
            //     {
            //         FILE *F;
            //         F = fopen("a_dx.gpd", "w");
            //     cst N = 10000;
            //     cdbl dx = 1.0 / N; 
            //     cdbl dy = 1.0 / N; 
            //     dbl andrianov_meta = 0.0;
            //     // arr<arr<dbl, N+1>, N+1> a_res;
            //
            //     auto solv_cos = [] (cdbl px, cdbl py){
            //         cdbl r = sqrt(px*px+py*py);
            //         return std::abs(r) > 1e-10 ? (px/r) : 0.0;
            //     };
            //
            //     // for (st i = 0; i < N+1; ++i)
            //     // {
            //     //     for (st j = 0; j < N+1; ++j)
            //     //     {
            //     //         cdbl X = -0.5 + dx * i;
            //     //         cdbl Y = -0.5 + dy * j;
            //     //         cdbl R = sqrt(X*X+Y*Y);
            //     //         cdbl material = ((R < R_in) ? 1.0e5 : 1.0);
            //     //         a_res[i][j] = solve_andrianov(dealii::Point<2>(X, Y))*solv_cos(X, Y);
            //     //         // printf("a_res %f\n", a_res[i][j]);
            //     //     };
            //     // };
            //
            //     cdbl deriver = (N+1)*(N+1);
            //     dbl temp_sum = 0.0;
            //     printf("deriver %f\n", deriver);
            //     for (st i = 0; i < N+1; ++i)
            //     {
            //         for (st j = 0; j < N+1; ++j)
            //         {
            //             cdbl X = -0.5 + dx * i;
            //             cdbl Y = -0.5 + dy * j;
            //             cdbl R = sqrt(X*X+Y*Y);
            //             cdbl material = ((R < R_in) ? lambda : 1.0);
            //             // cdbl cos_1 = std::abs(R) > 1e-5 ? ((X+dx/2.0)/R) : 0.0;
            //             // cdbl cos_2 = std::abs(R) > 1e-5 ? ((X-dx/2.0)/R) : 0.0;
            //             // auto solv_cos = [] (cdbl px, cdbl py){
            //             //     cdbl r = sqrt(px*px+py*py);
            //             //     return std::abs(r) > 1e-5 ? (px/r) : 0.0;
            //             // };
            //             // cdbl theta_1 = ((std::abs(R) > 1e-5) ? acos(X / R) : 0.0);
            //             // cdbl theta_2 = ((std::abs(R) > 1e-5) ? acos((X+dx) / R) : 0.0);
            //             // printf("%f %f %f %f %f\n", theta_1, theta_2, acos((X+dx) / R), );
            //
            //             cdbl x1 = X;
            //             cdbl x2 = X+dx;
            //             cdbl x3 = X;
            //             cdbl x4 = X+dx;
            //
            //             cdbl y1 = Y;
            //             cdbl y2 = Y;
            //             cdbl y3 = Y+dy;
            //             cdbl y4 = Y+dy;
            //
            //             // cdbl f1 = a_res[i][j];
            //             // cdbl f2 = a_res[i+1][j];
            //             // cdbl f3 = a_res[i][j+1];
            //             // cdbl f4 = a_res[i+1][j+1];
            //             cdbl f1 = solve_andrianov(dealii::Point<2>(X, Y))*solv_cos(X, Y);
            //             cdbl f2 = solve_andrianov(dealii::Point<2>(X+dx, Y))*solv_cos(X+dx, Y);
            //             cdbl f3 = solve_andrianov(dealii::Point<2>(X, Y+dy))*solv_cos(X, Y+dy);
            //             cdbl f4 = solve_andrianov(dealii::Point<2>(X+dx, Y+dy))*solv_cos(X+dx, Y+dy);
            //
            //             cdbl b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            //             // cdbl c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            //             cdbl d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
            //
            //             cdbl grad = b + d * Y;
            //
            //             // andrianov_meta += 
            //             temp_sum +=
            //                 (material * grad);// / deriver;
            //             if (std::abs(temp_sum) > deriver)
            //             {
            //                 andrianov_meta += temp_sum / deriver;
            //                 temp_sum = 0.0;
            //             };
            //             // material * 
            //             //     (solve_andrianov(dealii::Point<2>(X+dx/2.0, Y))*solv_cos(X+dx/2.0, Y) - 
            //             //     solve_andrianov(dealii::Point<2>(X-dx/2.0, Y))*solv_cos(X-dx/2.0, Y)) / dx; 
            //             // fprintf(F,"%f %f %f\n", X+0.5, Y+0.5, 
            //             //     //     material *
            //             //     // (solve_andrianov(dealii::Point<2>(X+dx/2.0, Y))*cos_1 - 
            //             //     // solve_andrianov(dealii::Point<2>(X-dx/2.0, Y))*cos_2) / dx); 
            //             //     solve_andrianov(dealii::Point<2>(X+dx/2.0, Y))*cos_1);
            //             // fprintf(F,"%f %f %f\n", X+0.5, Y+0.5, a_res[i][j]);
            //             // fprintf(F,"%f %f %f\n", X+0.5, Y+0.5, material * grad);
            //             // printf("%f %f %f %f\n", solve_andrianov(dealii::Point<2>(X+dx/2.0, Y)),
            //             //         solve_andrianov(dealii::Point<2>(X-dx/2.0, Y)),
            //             //         andrianov_meta, material);
            //         };
            //     };
            //     andrianov_meta += temp_sum / deriver;
            //     // andrianov_meta /= (N+1)*(N+1);
            //     printf("andrianov_meta %f %f\n", andrianov_meta, andrianov_meta + meta_coef[y][y]);
            //     printf("%f %f\n", temp_sum, deriver);
            //         fclose(F);
            //     };
            //     // fprintf(F, "%f %f %f\n", size*size, meta_coef[x][x], meta_coef[y][y]);
            //     // puts("111111111");
            //     // size+=0.05;
            //     // puts("2222222");
            //     // printf("%f %f\n", size, size*size);
            };
                // fclose(F);
        };
    };


ATools::FourthOrderTensor solve_elastic_problem_on_cell_3d_and_meta_coef_return ()
{
    {
        enum {x, y, z};
        Domain<3> domain;
        {
            set_cylinder(domain.grid, 0.3, z, 4);
            // set_ball(domain.grid, 0.4, 4);
                // set_rect_3d(domain.grid,
                //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
                //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
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

        arr<str, 6> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd", "move_xz.gpd", "move_yz.gpd"};
        for (st i = 0; i < 6; ++i)
        {
            // EPTools ::print_move<3> (slae.solution[i], domain.dof_handler, vr[i]);
            EPTools ::print_move_slice (slae.solution[i], domain.dof_handler, vr[i], z, 0.5);
        };
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice.gpd", z, 0.5);

        return OnCell::calculate_meta_coefficients_3d_elastic<3> (
                domain.dof_handler, slae, element_matrix.C);
    };
};

    void solve_elastic_problem (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
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
                std::vector< dealii::Point< 2 > > v (4);

                v[0] = dealii::Point<2>(0.0, 0.0);
                v[1] = dealii::Point<2>(1.0, 0.0);
                v[2] = dealii::Point<2>(1.0, 1.0);
                v[3] = dealii::Point<2>(0.0, 1.0);

                std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());

                c[0].vertices[0] = 0; 
                c[0].vertices[1] = 1; 
                c[0].vertices[2] = 3;
                c[0].vertices[3] = 2;
                c[0].material_id = 0; 

                dealii::SubCellData b;

                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 0;
                    cell.vertices[1] = 3;
                    cell.boundary_id = 1;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 0;
                    cell.vertices[1] = 1;
                    cell.boundary_id = 3;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 1;
                    cell.vertices[1] = 2;
                    cell.boundary_id = 2;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 3;
                    cell.vertices[1] = 2;
                    cell.boundary_id = 4;
                    b.boundary_lines .push_back (cell);
                };
                // // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 0});
                // // b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 2});
                // // b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
                // // b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 2});
                //
                // // dealii::GridReordering<2> ::reorder_cells (c);
                // // domain.grid .create_triangulation_compatibility (v, c, b);
                // domain.grid .create_triangulation (v, c, b);

                // domain.grid.refine_global(5);
                //
                // // dealii::GridGenerator::hyper_cube(domain.grid);
                // // domain.grid.refine_global(2);

                    vec<prmt::Point<2>> border;
                    vec<st> type_border;
                    // give_rectangle(border, 2,
                    //         prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                    give_rectangle_with_border_condition(
                            border,
                            type_border,
                            arr<st, 4>{1,3,2,4},
                            1,
                            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                    // for (auto i : type_border)
                    //     printf("type %d\n", i);
//                         vec<prmt::LoopCondition<2>> loop_border;
//                         // give_rectangle_for_loop_borders(border, loop_border, 8,
//                         //         prmt::Point<2>(0., 0.), prmt::Point<2>(1., 1.));
                    // vec<prmt::Point<2>> inclusion;
                    vec<vec<prmt::Point<2>>> inclusion(2);
                    cdbl radius = 0.25;
                    cdbl radius_2 = 0.255;
                    dealii::Point<2> center (0.5, 0.5);
                    give_circ(inclusion[0], 32, radius, prmt::Point<2>(center));
                    give_circ(inclusion[1], 40, radius_2, prmt::Point<2>(center));
                    // give_rectangle(inclusion, 1,
                    //         prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.75, 0.75));
                    // give_circ(inclusion, 20, radius, prmt::Point<2>(0.5, 0.5));
                    // vec<vec<prmt::Point<2>>> inclusion(2);
                    // give_rectangle(inclusion[0], 1,
                    //         prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.45, 0.45));
                    // give_rectangle(inclusion[1], 1,
                    //         prmt::Point<2>(0.65, 0.65), prmt::Point<2>(0.75, 0.75));
//                         give_crack<t_rounded_tip, 1>(inclusion, 30);
// 
                    ::set_grid(domain.grid, border, inclusion, type_border);
                    // dealii::GridGenerator ::hyper_cube (domain.grid, 0.0, 1.0);
                    // domain.grid.refine_global(4);
                    {
                        // dealii::Point<2> center (0.5, 0.5);
                        dealii::Triangulation<2>::active_cell_iterator
                            cell = domain.grid .begin_active(),
                                 end_cell = domain.grid .end();
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

                            // printf("%f %f\n", midle_p(0), midle_p(1));

                            if (center.distance(midle_p) < radius_2)
                            {
                                cell->set_material_id(2);
                                //                puts("adf");
                            }
                            else
                                cell->set_material_id(0);
                        };
                    };
                    {
                        // dealii::Point<2> center (0.5, 0.5);
                        dealii::Triangulation<2>::active_cell_iterator
                            cell = domain.grid .begin_active(),
                                 end_cell = domain.grid .end();
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

                            // printf("%f %f\n", midle_p(0), midle_p(1));

                            if (center.distance(midle_p) < radius)
                            {
                                cell->set_material_id(1);
                                //                puts("adf");
                            }
                            // else
                            //     cell->set_material_id(0);
                        };
                    };
                // domain.grid.refine_global(1);
            };
            dealii::FESystem<2,2> fe 
                (dealii::FE_Q<2,2>(1), 2);
            domain.dof_init (fe);

            SystemsLinearAlgebraicEquations slae;
            ATools ::trivial_prepare_system_equations (slae, domain);

            LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
            element_matrix.C .resize (3);
            EPTools ::set_isotropic_elascity{yung : 20.0, puasson : 0.25}(element_matrix.C[0]);
            // EPTools ::set_isotropic_elascity{yung : 0.0, puasson : 0.00}(element_matrix.C[1]);
            EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);
            EPTools ::set_isotropic_elascity{yung : 20.0, puasson : 0.25}(element_matrix.C[2]);
            // auto meta_coef = solve_elastic_problem_on_cell_3d_and_meta_coef_return ();
            ATools::FourthOrderTensor meta_coef;
            std::ifstream in ("cell/meta_coef.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &meta_coef, sizeof meta_coef);
            in.close ();
            // for (st i = 0; i < 3; ++i)
            // {
            //     for (st j = 0; j < 3; ++j)
            //     {
            //         for (st k = 0; k < 3; ++k)
            //         {
            //             for (st l = 0; l < 3; ++l)
            //             {
            //                 element_matrix.C[0][i][j][k][l] = meta_coef[i][j][k][l];
            //                 element_matrix.C[1][i][j][k][l] = 0.0;
            //                 // element_matrix.C[1][i][j][k][l] = meta_coef[i][j][k][l];
            //                 element_matrix.C[2][i][j][k][l] = meta_coef[i][j][k][l];
            //                 // element_matrix.C[0][2-i][2-j][2-k][2-l] = meta_coef[i][j][k][l];
            //                 // element_matrix.C[1][2-i][2-j][2-k][2-l] = 0.0;
            //                 // // element_matrix.C[1][i][j][k][l] = meta_coef[i][j][k][l];
            //                 // element_matrix.C[2][2-i][2-j][2-k][2-l] = meta_coef[i][j][k][l];
            //             };
            //         };
            //     };
            // };
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

                if (std::abs(element_matrix.C[0][im][in][jm][jn]) > 0.0000001)
                    printf("\x1B[31m%f\x1B[0m   ", 
                            element_matrix.C[0][im][in][jm][jn]);
                else
                    printf("%f   ", 
                            element_matrix.C[0][im][in][jm][jn]);
            };
            for (size_t i = 0; i < 2; ++i)
                printf("\n");
        };

        {
            std::ofstream out ("hole/solution_hole_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution.size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };

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

            vec<BoundaryValueVector<2>> bound (4);
            // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{p(0), 0.0};};
            // bound[0].boundary_id   = 0;
            // bound[0].boundary_type = TBV::Dirichlet;
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
            bound[2].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -1.0};};
            bound[2].boundary_id   = 3;
            bound[2].boundary_type = TBV::Neumann;
            bound[3].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 1.0};};
            bound[3].boundary_id   = 4;
            bound[3].boundary_type = TBV::Neumann;
            bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
            bound[0].boundary_id   = 1;
            bound[0].boundary_type = TBV::Neumann;
            bound[1].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
            bound[1].boundary_id   = 2;
            bound[1].boundary_type = TBV::Neumann;

            for (auto b : bound)
                ATools ::apply_boundary_value_vector<2> (b) .to_slae (slae, domain);

            dealii::SolverControl solver_control (50000, 1e-12);
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
            EPTools ::print_elastic_deformation (slae.solution, domain.dof_handler, "deform.gpd");
            EPTools ::print_elastic_deformation_mean (slae.solution, domain.dof_handler, "deform_mean.gpd");
            EPTools ::print_elastic_deformation_mean_other (slae.solution, domain.dof_handler, "deform_other_2d.gpd");
            EPTools ::print_elastic_deformation_2 (slae.solution, domain.dof_handler, "deform_2.gpd");
            // EPTools ::print_elastic_deformation_quad (slae.solution, domain.dof_handler, "deform_quad.gpd");
            // EPTools ::print_elastic_stress_quad (slae.solution, domain.dof_handler, 
            //         element_matrix.C[0], "stress_quad.gpd");
            EPTools ::print_elastic_stress (slae.solution, domain.dof_handler, 
                    element_matrix.C[0], "stress.gpd");
            EPTools ::print_elastic_mean_micro_stress (slae.solution, domain.dof_handler,
                    element_matrix.C[0], "mean_micro_stress.gpd", 0.01);
            EPTools ::print_coor_bin<2> (domain.dof_handler, "hole/coor_hole.bin");
        };
    };

    void solve_elastic_problem (cst flag, cst refine, const str f_name_cell, const str f_name)
    {
        if (flag)
        {
            enum {x, y, z};

            if (access(("hole/"+f_name).c_str(), 0))
            {
                mkdir(("hole/"+f_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            };
        
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
                std::vector< dealii::Point< 2 > > v (4);

                v[0] = dealii::Point<2>(0.0, 0.0);
                v[1] = dealii::Point<2>(1.0, 0.0);
                v[2] = dealii::Point<2>(1.0, 1.0);
                v[3] = dealii::Point<2>(0.0, 1.0);

                std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());

                c[0].vertices[0] = 0; 
                c[0].vertices[1] = 1; 
                c[0].vertices[2] = 3;
                c[0].vertices[3] = 2;
                c[0].material_id = 0; 

                dealii::SubCellData b;

                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 0;
                    cell.vertices[1] = 3;
                    cell.boundary_id = 1;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 0;
                    cell.vertices[1] = 1;
                    cell.boundary_id = 3;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 1;
                    cell.vertices[1] = 2;
                    cell.boundary_id = 2;
                    b.boundary_lines .push_back (cell);
                };
                {
                    dealii::CellData<1> cell;
                    cell.vertices[0] = 3;
                    cell.vertices[1] = 2;
                    cell.boundary_id = 4;
                    b.boundary_lines .push_back (cell);
                };
                // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 0});
                // b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 2});
                // b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
                // b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 2});

                // dealii::GridReordering<2> ::reorder_cells (c);
                // domain.grid .create_triangulation_compatibility (v, c, b);
                domain.grid .create_triangulation (v, c, b);

                domain.grid.refine_global(refine);

                // dealii::GridGenerator::hyper_cube(domain.grid);
                // domain.grid.refine_global(2);

                    vec<prmt::Point<2>> border;
                    vec<st> type_border;
                    // give_rectangle(border, 2,
                    //         prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                    give_rectangle_with_border_condition(
                            border,
                            type_border,
                            arr<st, 4>{1,3,2,4},
                            1,
                            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                    // for (auto i : type_border)
                    //     printf("type %d\n", i);
//                         vec<prmt::LoopCondition<2>> loop_border;
//                         // give_rectangle_for_loop_borders(border, loop_border, 8,
//                         //         prmt::Point<2>(0., 0.), prmt::Point<2>(1., 1.));
                    // vec<prmt::Point<2>> inclusion;
                    vec<vec<prmt::Point<2>>> inclusion(2);
                    cdbl radius = 0.05;
                    cdbl radius_2 = 0.055;
                    dealii::Point<2> center (0.5, 0.5);
                    give_circ(inclusion[0], 32, radius, prmt::Point<2>(center));
                    give_circ(inclusion[1], 40, radius_2, prmt::Point<2>(center));
                    // give_rectangle(inclusion, 1,
                    //         prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.75, 0.75));
                    // give_circ(inclusion, 20, radius, prmt::Point<2>(0.5, 0.5));
                    // vec<vec<prmt::Point<2>>> inclusion(2);
                    // give_rectangle(inclusion[0], 1,
                    //         prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.45, 0.45));
                    // give_rectangle(inclusion[1], 1,
                    //         prmt::Point<2>(0.65, 0.65), prmt::Point<2>(0.75, 0.75));
//                         give_crack<t_rounded_tip, 1>(inclusion, 30);
// 
                    // ::set_grid(domain.grid, border, inclusion, type_border);
                    // dealii::GridGenerator ::hyper_cube (domain.grid, 0.0, 1.0);
                    // domain.grid.refine_global(4);
                    {
                        // dealii::Point<2> center (0.5, 0.5);
                        dealii::Triangulation<2>::active_cell_iterator
                            cell = domain.grid .begin_active(),
                                 end_cell = domain.grid .end();
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

                            // printf("%f %f\n", midle_p(0), midle_p(1));

                            if (center.distance(midle_p) < radius_2)
                            {
                                cell->set_material_id(2);
                                //                puts("adf");
                            }
                            else
                                cell->set_material_id(0);
                        };
                    };
                    {
                        // dealii::Point<2> center (0.5, 0.5);
                        dealii::Triangulation<2>::active_cell_iterator
                            cell = domain.grid .begin_active(),
                                 end_cell = domain.grid .end();
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

                            // printf("%f %f\n", midle_p(0), midle_p(1));

                            if (center.distance(midle_p) < radius)
                            {
                                cell->set_material_id(1);
                                //                puts("adf");
                            }
                            // else
                            //     cell->set_material_id(0);
                        };
                    };
                domain.grid.refine_global(1);
            };
            dealii::FESystem<2,2> fe 
                (dealii::FE_Q<2,2>(1), 2);
            domain.dof_init (fe);

            SystemsLinearAlgebraicEquations slae;
            ATools ::trivial_prepare_system_equations (slae, domain);

            LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
            element_matrix.C .resize (3);
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
            // // EPTools ::set_isotropic_elascity{yung : 0.0, puasson : 0.00}(element_matrix.C[1]);
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[2]);
            // // auto meta_coef = solve_elastic_problem_on_cell_3d_and_meta_coef_return ();
            ATools::FourthOrderTensor meta_coef;
            std::ifstream in ("cell/"+f_name_cell+"/meta_coef.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &meta_coef, sizeof meta_coef);
            in.close ();
            for (st i = 0; i < 3; ++i)
            {
                for (st j = 0; j < 3; ++j)
                {
                    for (st k = 0; k < 3; ++k)
                    {
                        for (st l = 0; l < 3; ++l)
                        {
                            element_matrix.C[0][i][j][k][l] = meta_coef[i][j][k][l];
                            element_matrix.C[1][i][j][k][l] = 0.0;
                            // element_matrix.C[1][i][j][k][l] = meta_coef[i][j][k][l];
                            element_matrix.C[2][i][j][k][l] = meta_coef[i][j][k][l];
                            // element_matrix.C[0][2-i][2-j][2-k][2-l] = meta_coef[i][j][k][l];
                            // element_matrix.C[1][2-i][2-j][2-k][2-l] = 0.0;
                            // // element_matrix.C[1][i][j][k][l] = meta_coef[i][j][k][l];
                            // element_matrix.C[2][2-i][2-j][2-k][2-l] = meta_coef[i][j][k][l];
                        };
                    };
                };
            };
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

                if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
                    printf("\x1B[31m%f\x1B[0m   ", 
                            element_matrix.C[0][im][in][jm][jn]);
                else
                    printf("%f   ", 
                            element_matrix.C[0][im][in][jm][jn]);
            };
            for (size_t i = 0; i < 2; ++i)
                printf("\n");
        };

        {
            std::ofstream out ("hole/"+f_name+"/solution_hole_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution.size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };

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

            vec<BoundaryValueVector<2>> bound (4);
            // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{p(0), 0.0};};
            // bound[0].boundary_id   = 0;
            // bound[0].boundary_type = TBV::Dirichlet;
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
            bound[2].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -1.0};};
            bound[2].boundary_id   = 3;
            bound[2].boundary_type = TBV::Neumann;
            bound[3].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 1.0};};
            bound[3].boundary_id   = 4;
            bound[3].boundary_type = TBV::Neumann;
            bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
            bound[0].boundary_id   = 1;
            bound[0].boundary_type = TBV::Neumann;
            bound[1].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
            bound[1].boundary_id   = 2;
            bound[1].boundary_type = TBV::Neumann;

            for (auto b : bound)
                ATools ::apply_boundary_value_vector<2> (b) .to_slae (slae, domain);

            dealii::SolverControl solver_control (50000, 1e-12);
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
            EPTools ::print_elastic_deformation (slae.solution, domain.dof_handler, "deform.gpd", f_name);
            EPTools ::print_elastic_deformation_mean (slae.solution, domain.dof_handler, "deform_mean.gpd");
            EPTools ::print_elastic_deformation_mean_other (slae.solution, domain.dof_handler, "deform_other_2d.gpd");
            EPTools ::print_elastic_deformation_2 (slae.solution, domain.dof_handler, "deform_2.gpd", f_name);
            // EPTools ::print_elastic_deformation_quad (slae.solution, domain.dof_handler, "deform_quad.gpd");
            // EPTools ::print_elastic_stress_quad (slae.solution, domain.dof_handler, 
            //         element_matrix.C[0], "stress_quad.gpd");
            EPTools ::print_elastic_stress (slae.solution, domain.dof_handler, 
                    element_matrix.C[0], "stress.gpd", f_name);
            EPTools ::print_elastic_mean_micro_stress (slae.solution, domain.dof_handler,
                    element_matrix.C[0], "mean_micro_stress.gpd", 0.01);
            EPTools ::print_coor_bin<2> (domain.dof_handler, "hole/"+f_name+"/coor_hole.bin");
        };
    };

void solve_plate_with_hole_problem (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<2> domain;
        {
            dealii::GridGenerator::hyper_cube_with_cylindrical_holei<2>(domain.grid);
            {
                std::ofstream out ("grid-igor.eps");
                dealii::GridOut grid_out;
                grid_out.write_eps (domain.grid, out);
            };
        };
        // dealii::FESystem<2,2> fe 
        //     (dealii::FE_Q<2,2>(1), 2);
        // domain.dof_init (fe);
        //
        // SystemsLinearAlgebraicEquations slae;
        // ATools ::trivial_prepare_system_equations (slae, domain);
        //
        // LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
        // element_matrix.C .resize (1);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
        // ATools::FourthOrderTensor meta_coef;
        // std::ifstream in ("cell/meta_coef.bin", std::ios::in | std::ios::binary);
        // in.read ((char *) &meta_coef, sizeof meta_coef);
        // in.close ();
        // // for (st i = 0; i < 3; ++i)
        // // {
        // //     for (st j = 0; j < 3; ++j)
        // //     {
        // //         for (st k = 0; k < 3; ++k)
        // //         {
        // //             for (st l = 0; l < 3; ++l)
        // //             {
        // // element_matrix.C[0][i][j][k][l] = meta_coef[i][j][k][l];
        // //             };
        // //         };
        // //     };
        // // };
        // for (size_t i = 0; i < 9; ++i)
        // {
        //     uint8_t im = i / (2 + 1);
        //     uint8_t in = i % (2 + 1);
        //
        //     for (size_t j = 0; j < 9; ++j)
        //     {
        //         uint8_t jm = j / (2 + 1);
        //         uint8_t jn = j % (2 + 1);
        //
        //         if (std::abs(element_matrix.C[0][im][in][jm][jn]) > 0.0000001)
        //             printf("\x1B[31m%f\x1B[0m   ", 
        //                     element_matrix.C[0][im][in][jm][jn]);
        //         else
        //             printf("%f   ", 
        //                     element_matrix.C[0][im][in][jm][jn]);
        //     };
        //     for (size_t i = 0; i < 2; ++i)
        //         printf("\n");
        // };
        //
        //
        // arr<std::function<dbl (const dealii::Point<2>&)>, 2> func {
        //     // [=] (const dealii::Point<2>) {return -2.0*abld;},
        //     [] (const dealii::Point<2>) {return 0.0;},
        //     [] (const dealii::Point<2>) {return 0.0;}
        // };
        // // auto func = [] (dealii::Point<2>) {return arr<dbl, 2>{-2.0, 0.0};};
        // SourceVector<2> element_rhsv (func, domain.dof_handler.get_fe());
        //
        // Assembler ::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
        // Assembler ::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);
        //
        // vec<BoundaryValueVector<2>> bound (4);
        // bound[2].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -1.0};};
        // bound[2].boundary_id   = 3;
        // bound[2].boundary_type = TBV::Neumann;
        // // bound[0].boundary_type = TBV::Dirichlet;
        // bound[3].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 1.0};};
        // bound[3].boundary_id   = 4;
        // bound[3].boundary_type = TBV::Neumann;
        // bound[0].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[0].boundary_id   = 1;
        // bound[0].boundary_type = TBV::Neumann;
        // bound[1].function      = [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};};
        // bound[1].boundary_id   = 2;
        // bound[1].boundary_type = TBV::Neumann;
        //
        // for (auto b : bound)
        //     ATools ::apply_boundary_value_vector<2> (b) .to_slae (slae, domain);
        //
        // dealii::SolverControl solver_control (50000, 1e-12);
        // dealii::SolverCG<> solver (solver_control);
        // solver.solve (
        //         slae.matrix,
        //         slae.solution,
        //         slae.rhsv
        //         ,dealii::PreconditionIdentity()
        //         );
        // //
        // // EPTools ::print_move<2> (indexes, domain.dof_handler, "move.gpd");
        // EPTools ::print_move<2> (slae.solution, domain.dof_handler, "move.gpd");
        // // EPTools ::print_elastic_deformation (slae.solution, domain.dof_handler, "deform.gpd");
        // // EPTools ::print_elastic_deformation_mean (slae.solution, domain.dof_handler, "deform_mean.gpd");
        // // EPTools ::print_elastic_deformation_mean_other (slae.solution, domain.dof_handler, "deform_other_2d.gpd");
        // // EPTools ::print_elastic_deformation_2 (slae.solution, domain.dof_handler, "deform_2.gpd");
        // // // EPTools ::print_elastic_deformation_quad (slae.solution, domain.dof_handler, "deform_quad.gpd");
        // // // EPTools ::print_elastic_stress_quad (slae.solution, domain.dof_handler, 
        // // //         element_matrix.C[0], "stress_quad.gpd");
        // // EPTools ::print_elastic_stress (slae.solution, domain.dof_handler, 
        // //         element_matrix.C[0], "stress.gpd");
        // // EPTools ::print_elastic_mean_micro_stress (slae.solution, domain.dof_handler,
        // //         element_matrix.C[0], "mean_micro_stress.gpd", 0.01);
        // // EPTools ::print_coor_bin<2> (domain.dof_handler, "hole/coor_hole.bin");
    };
};

    void solve_elastic_problem_on_cell (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
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
                    // dealii::GridGenerator ::hyper_cube (domain.grid, 0.0, 1.0);
                    // domain.grid .refine_global (2);
                //     auto map = dealii::GridTools ::get_all_vertices_at_boundary (domain.grid);
                //     auto ver = domain.grid.get_vertices();
                //     // auto map = dealii::GridTools ::diameter (domain.grid);
                //     // std::cout << map.at(6) << std::endl;
                //     for (auto &&m : map)
                //         std::cout << m.first << " " << m.second << " : " << ver[m.first] << std::endl;
                //
                //     dealii::FESystem<2,2> fe (dealii::FE_Q<2,2>(1), 2);
                //     domain.dof_init (fe);
                // OnCell::SystemsLinearAlgebraicEquations<4> slae;
                // OnCell::BlackOnWhiteSubstituter bows;
                //
                // const bool vector_type = 1;
                // OnCell::prepare_system_equations_alternate<2, 2, 4> (slae, bows, domain);
                //     // domain.dof_handler.vertex_dofs;
                //
                // exit(1);
                //    
                //     auto cell = domain.dof_handler.begin_active();
                //     auto endc = domain.dof_handler.end();
                //     for (; cell != endc; ++cell)
                //     {
                //         auto dof  = cell -> vertex_dof_index(0,0);
                //         auto vert = cell -> vertex_index(0);
                //         if (map.find(vert) != map.end())
                //         std::cout << dof << " " << vert << " " << map[vert] << std::endl;
                //     };
                //     dealii::DynamicSparsityPattern dsp(domain.dof_handler.n_dofs());
                //     dealii::DoFTools::make_sparsity_pattern (domain.dof_handler, dsp);
                //     dealii::SparsityPattern sparsity;
                //     sparsity.copy_from(dsp);
                //     // dealii::DoFTools ::make_sparsity_pattern (
                //     //         domain.dof_handler, c_sparsity);
                //     dsp .add (1,10);
                //     std::cout << dsp.exists(1,10) << std::endl;
                //     auto mapd = dealii::GridTools ::get_all_vertices_at_boundary (domain.dof_handler.get_tria());
                //     auto verd = domain.dof_handler.get_tria().get_vertices();
                //     // puts ("");
                //     // for (auto &&v : ver)
                //     //     std::cout << v << std::endl;
                //     // for (st i = 0; i < map.size(); ++i)
                //     // {
                //     //     std::cout << map[i] << " : " << ver[i] << std::endl;
                //     // };
                //     exit(1);
                //     // const size_t material_id[4][4] =
                //     // {
                //     //     {0, 0, 0, 0},
                //     //     {0, 1, 1, 0},
                //     //     {0, 1, 1, 0},
                //     //     {0, 0, 0, 0}
                //     // };
                //     // const double dot[5] = 
                //     // {
                //     //     (0.0),
                //     //     (0.5 - size / 2.0),
                //     //     (0.5),
                //     //     (0.5 + size / 2.0),
                //     //     (1.0)
                //     // };
                //     // ::set_tria <5> (domain.grid, dot, material_id);
                //     // ::set_hexagon_grid_pure (domain.grid, 1.0, size);
                //     // set_circ(domain.grid, 0.475, 4);
                //     // set_circ(domain.grid, 0.25, 7);
                //     // domain.grid .refine_global (3);
                //     // vec<prmt::Point<2>> border;
                //     // vec<prmt::Point<2>> inclusion;
                //     // cdbl radius = 0.25;
                //     // dealii::Point<2> center (0.5, 0.5);
                //     // // give_circ(inclusion, 16, radius, prmt::Point<2>(center));
                //     // give_rectangle(inclusion, 16,
                //     //         prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.75, 0.75));
                //     // give_rectangle(border, 16,
                //     //         prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                //     // ::set_grid(domain.grid, border, inclusion);
                //
                    vec<prmt::Point<2>> border;
                    vec<st> type_border;
                    give_rectangle_with_border_condition(
                            border,
                            type_border,
                            arr<st, 4>{1,3,2,4},
                            10,
                            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
                    vec<vec<prmt::Point<2>>> inclusion(1);
                    cdbl radius = 0.25;
                //     // cdbl radius_2 = 0.255;
                    dealii::Point<2> center (0.5, 0.5);
                    give_circ(inclusion[0], 62, radius, prmt::Point<2>(center));
                    // give_rectangle(inclusion[0], 10, prmt::Point<2>(0.0, 0.0), prmt::Point<2>(0.5, 0.5));
                    // give_circ(inclusion[1], 40, radius_2, prmt::Point<2>(center));
                    ::set_grid(domain.grid, border, inclusion, type_border);
                    // ::set_hexagon_grid_pure (domain.grid, 1.0, 0.5);
                    // domain.grid .refine_global (3);

                    {
                        std::ofstream out ("grid-igor.eps");
                        dealii::GridOut grid_out;
                        grid_out.write_eps (domain.grid, out);
                    };
                    {
                        // dealii::Point<2> center (0.5, 0.5);
                        dealii::Triangulation<2>::active_cell_iterator
                            cell = domain.grid .begin_active(),
                                 end_cell = domain.grid .end();
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

                            // printf("%f %f\n", midle_p(0), midle_p(1));

                            if (center.distance(midle_p) < radius)
                            {
                                cell->set_material_id(1);
                                               // puts("adf");
                            }
                            else
                            {
                                cell->set_material_id(0);
                                // puts("123");
                            };
                        };
                    };
                };
                dealii::FESystem<2,2> fe (dealii::FE_Q<2,2>(1), 2);
                domain.dof_init (fe);

                OnCell::SystemsLinearAlgebraicEquations<4> slae;
                OnCell::SystemsLinearAlgebraicEquations<4> slae_2;
                OnCell::BlackOnWhiteSubstituter bows;
                OnCell::BlackOnWhiteSubstituter bows_2;

                LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
                element_matrix.C .resize (2);
                EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
                EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(element_matrix.C[1]);
                // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
                // EPTools ::set_isotropic_elascity{yung : 0.0, puasson : 0.0}(element_matrix.C[1]);

                u8 dim = 2;

                const bool vector_type = 1;
                // OnCell::prepare_system_equations<vector_type> (slae, bows, domain);
                // OnCell::prepare_system_equations_alternate<2, 2, 4> (slae_2, bows_2, domain);
                OnCell::prepare_system_equations_alternate<2, 2, 4> (slae, bows, domain);
                // std::cout << bows.size << " " << bows_2.size << std::endl;
                // for (st i = 0; i < bows.size; ++i)
                // {
                //     std::cout << bows.white[i] << " " << bows.black[i] << std::endl;
                // };
                // puts("???????????????????");
                // for (st i = 0; i < bows_2.size; ++i)
                // {
                //     std::cout << bows_2.white[i] << " " << bows_2.black[i] << std::endl;
                // };
                // for (st i = 0; i < 18; ++i)
                // {
                //     for (st j = 0; j < 18; ++j)
                //     {
                //         std::cout << slae.sparsity_pattern.exists(i,j) << " ";
                //     };
                //     std::cout << std::endl;
                // };
                // std::cout << std::endl;
                // for (st i = 0; i < 18; ++i)
                // {
                //     for (st j = 0; j < 18; ++j)
                //     {
                //         std::cout << slae_2.sparsity_pattern.exists(i,j) << " ";
                //     };
                //     std::cout << std::endl;
                // };
                // std::cout << std::endl;
                // exit(1);
        //         for (st i = 0; i < bows.size; ++i)
        //         {
        //             std::cout << bows.white[i] << " " << bows.black[i] << " " << bows.subst(bows.black[i])<< std::endl;
        //         };
        // {
        //     std::ofstream f("/home/primat/tmp/sp1.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // {
        //     std::ofstream f("/home/primat/tmp/sp2.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae_2.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // exit(1);

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
                solve_heat_problem_on_cell_aka_torsion_rod<2> (
                        domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);

                arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
                for (st i = 0; i < 4; ++i)
                {
                    EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
                };
            EPTools ::print_elastic_stress (slae.solution[0], domain.dof_handler, 
                    element_matrix.C[0], "cell_stress_xx.gpd");
            EPTools ::print_elastic_deformation (slae.solution[0], domain.dof_handler, "cell_deform.gpd");
            EPTools ::print_elastic_deformation_mean (slae.solution[0], domain.dof_handler, "cell_deform_mean.gpd");

                auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
                        domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);
        {
        std::ofstream out ("meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

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
            std::cout << 
                newcoef[2][2][0][0]/newcoef[0][0][0][0] <<  " " <<
                newcoef[0][0][2][2]/newcoef[2][2][2][2] <<  " " <<
                newcoef[0][0][2][2]/newcoef[0][0][0][0] <<  " " <<
                newcoef[2][2][0][0]/newcoef[2][2][2][2]   
                << std::endl;
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
    };

    void solve_heat_conduction_nikola_problem (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
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

                // std::vector< dealii::Point< 2 > > v (6);

                // v[0]  = dealii::Point<2>(0.0, -0.5);
                // v[1]  = dealii::Point<2>(1.0, -0.5);
                // v[2]  = dealii::Point<2>(0.0, 0.0);
                // v[3]  = dealii::Point<2>(1.0, 0.0);
                // v[4]  = dealii::Point<2>(0.0, 0.5);
                // v[5]  = dealii::Point<2>(1.0, 0.5);
                // // v[0]  = dealii::Point<2>(0.0, 0.0);
                // // v[1]  = dealii::Point<2>(0.0, 1.0);
                // // v[2]  = dealii::Point<2>(0.5, 0.0);
                // // v[3]  = dealii::Point<2>(0.5, 1.0);
                // // v[4]  = dealii::Point<2>(1.0, 0.0);
                // // v[5]  = dealii::Point<2>(1.0, 1.0);

                // std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());
                // {
                //     dealii::CellData<2> tmp;
                //     tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.vertices[2]=3;tmp.vertices[3]=2;tmp.material_id=0;
                //     c .push_back (tmp);
                //     tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.vertices[2]=4;tmp.vertices[3]=2;tmp.material_id=1;
                //     c .push_back (tmp);
                // };
                // // c .push_back (dealii::CellData<2>{{0, 1, 3, 2}, 0});
                // // c .push_back (dealii::CellData<2>{{3, 5, 4, 2}, 1});

                // dealii::SubCellData b;
                // {
                //     dealii::CellData<1> tmp;
                //     tmp.vertices[0]=4;tmp.vertices[1]=2;tmp.boundary_id=0;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=2;tmp.vertices[1]=0;tmp.boundary_id=0;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.boundary_id=1;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=1;tmp.vertices[1]=3;tmp.boundary_id=2;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.boundary_id=2;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=5;tmp.vertices[1]=4;tmp.boundary_id=3;
                //     b.boundary_lines .push_back (tmp);
                // };
                // // b.boundary_lines .push_back (dealii::CellData<1>{4, 2, 0});
                // // b.boundary_lines .push_back (dealii::CellData<1>{2, 0, 0});
                // // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
                // // b.boundary_lines .push_back (dealii::CellData<1>{1, 3, 2});
                // // b.boundary_lines .push_back (dealii::CellData<1>{3, 5, 2});
                // // b.boundary_lines .push_back (dealii::CellData<1>{5, 4, 3});

                // dealii::GridReordering<2> ::reorder_cells (c);
                // domain.grid .create_triangulation_compatibility (v, c, b);

                // domain.grid .refine_global (4);
                
            // set_quadrate<2> (domain.grid, 
            //             0.0, 1.0/3.0, 2.0/3.0, 1.0, 
            //             -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
            //             4);
                // set_tube(domain.grid, str("grid-igor_ss8.40749_h1_rho0.4.msh"), 
                //         dealii::Point<2>(4.0,4.0), 1.0, 2.0, 1);
                dealii::GridIn<2> gridin;
                gridin.attach_triangulation(domain.grid);
                std::ifstream f("grid-igor_ss8.40749_h1_rho0.4.msh");
                gridin.read_msh(f);
                // set_quadrate<2> (domain.grid, 
                //         0.0, 1.0/3.0, 2.0/4.0, 1.0, 
                //         -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
                //         4);
                // set_hexagon_grid_pure(domain.grid, 100.0, 50.0);
                // domain.grid .refine_global (4);
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
            vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
            U[0][x] = [] (const dealii::Point<2> &p) {return 1.0;}; //Ux
            U[0][y] = [] (const dealii::Point<2> &p) {return 0.0;}; //Uy
            U[1][x] = [] (const dealii::Point<2> &p) {return 1.0;};
            U[1][y] = [] (const dealii::Point<2> &p) {return 0.0;};
            vec<typename Nikola::SourceScalar<2>::Func> tau(2);
            tau[0] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[1] = [] (const dealii::Point<2> &p) {return 0.0;};

            // T1.2
            // cdbl c0 = 0.5;
            // cdbl E = 1.0;
            // cdbl nu = 0.25;
            // cdbl mu = 0.4;
            // vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
            // // vec<arr<typename SourceScalarFeature<2>::Func, 2>> U(2);
            // // U[0][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));}; //Ux
            // // U[0][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*(p(0)-c0)*p(1);}; //Uy
            // // U[1][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));};
            // // U[1][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return mu*nu*(p(0)-c0)*p(1);};
            // // U[0][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;}; //Ux
            // // U[0][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;}; //Uy
            // // U[1][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;};
            // // U[1][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;};
            // U[0][x] = [] (const dealii::Point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;}; //Ux
            // U[0][y] = [] (const dealii::Point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);}; //Uy
            // U[1][x] = [] (const dealii::Point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;};
            // U[1][y] = [] (const dealii::Point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);};
            // vec<typename Nikola::SourceScalar<2>::Func> tau(2);
            // // vec<typename Nikola::SourceScalar<2>::Func> tau(2);
            // tau[0] = [E, c0] (const dealii::Point<2> &p) {return E*(p(0)-c0);};
            // tau[1] = [E, c0] (const dealii::Point<2> &p) {return E*(p(0)-c0);};
            // // tau[0] = [E, c0] (const dealii::Point<2> &p) {return 0.0;};
            // // tau[1] = [E, c0] (const dealii::Point<2> &p) {return 0.0;};
            // // tau[0] = [] (const dealii::Point<2> &p) {return 1.0*p(0);};
            // // tau[1] = [] (const dealii::Point<2> &p) {return 1.0*p(0);};
            // // tau[0] = [] (const dealii::Point<2> &p) {return -1+0.4*0.25*p(0)+0.25*p(0);};
            // // tau[1] = [] (const dealii::Point<2> &p) {return -1+0.4*0.25*p(0)+0.25*p(0);};
            //
            // // vec<arr<typename Nikola::SourceScalar<2>::Func, 2>> U(2);
            // // U[0][x] = [] (const dealii::Point<2> &p) {return 0.0;}; //Ux
            // // U[0][y] = [] (const dealii::Point<2> &p) {return 0.0;}; //Uy
            // // U[1][x] = [] (const dealii::Point<2> &p) {return 0.0;};
            // // U[1][y] = [] (const dealii::Point<2> &p) {return 0.0;};
            // // vec<typename Nikola::SourceScalar<2>::Func> tau(2);
            // // tau[0] = [] (const dealii::Point<2> &p) {return -2.0;};
            // // tau[1] = [] (const dealii::Point<2> &p) {return -2.0;};
            
            Nikola::SourceScalar<2> element_rhsv (U, tau, domain.dof_handler.get_fe());
            // auto func = [] (const dealii::Point<2> p) {return -1.0*(p(0)-0.5);};
            // SourceScalar<2> element_rhsv (func, domain.dof_handler.get_fe());
            SourceScalarFeature<2> element_rhsv1 (U, domain.dof_handler.get_fe());
            SourceScalarPolyMaterials<2> element_rhsv2 (tau, domain.dof_handler.get_fe());

            Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
            // Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);
            Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv1, domain.dof_handler);
            Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv2, domain.dof_handler);

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

            // HCPTools ::print_temperature<2> (slae.rhsv, domain.dof_handler, "b");
            // dbl sum = 0.0;
            // for (st i = 0; i < slae.rhsv.size(); ++i)
            // {
            //     sum += slae.rhsv(i);
            // };
            // printf("Integral %f\n", sum);

            dealii::SolverControl solver_control (100000, 1e-12);
            dealii::SolverCG<> solver (solver_control);
            solver.solve (
                    slae.matrix,
                    slae.solution,
                    slae.rhsv
                    ,dealii::PreconditionIdentity()
                    );

            dbl Integral = 0.0;
            dbl area_of_domain = 0.0;
            {
                dealii::QGauss<2>  quadrature_formula(2);

                dealii::FEValues<2> fe_values (domain.dof_handler.get_fe(), quadrature_formula,
                        dealii::update_quadrature_points | dealii::update_JxW_values |
                        dealii::update_values);

                cst n_q_points = quadrature_formula.size();


                auto cell = domain.dof_handler.begin_active();
                auto endc = domain.dof_handler.end();
                for (; cell != endc; ++cell)
                {
                    fe_values .reinit (cell);

                    dbl area_of_cell = 0.0;
                    for (st q_point = 0; q_point < n_q_points; ++q_point)
                        area_of_cell += fe_values.JxW(q_point);

                    dealii::Point<2> c_point(
                            (cell->vertex(0)(0) +
                             cell->vertex(1)(0) +
                             cell->vertex(2)(0) +
                             cell->vertex(3)(0)) / 4.0,
                            (cell->vertex(0)(1) +
                             cell->vertex(1)(1) +
                             cell->vertex(2)(1) +
                             cell->vertex(3)(1)) / 4.0);
                    Integral += get_value<2> (cell, slae.solution, c_point) * area_of_cell; 

                    area_of_domain += area_of_cell;
                };
            };
            printf("%f %f %f\n", Integral, area_of_domain, Integral / area_of_domain);

            for (st i = 0; i < slae.solution.size(); ++i)
            {
                slae.solution(i) -= Integral;
            };
            dealii::Vector<dbl> uber(slae.solution.size());
            dealii::Vector<dbl> diff(slae.solution.size());
            for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
            {
                for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
                {
                    dbl indx = cell->vertex_dof_index(i, 0);
                    uber(indx) = uber_function(cell->vertex(i), 200);
                    diff(indx) = 
                        // std::abs(uber(indx) - slae.solution(indx));
                     uber(indx) - slae.solution(indx);
                };
            };

            HCPTools ::print_temperature<2> (slae.solution, domain.dof_handler, "temperature-4.gpd");
            HCPTools ::print_temperature<2> (uber, domain.dof_handler, "uber.gpd");
            HCPTools ::print_temperature<2> (diff, domain.dof_handler, "uber-diff-2.gpd");
            HCPTools ::print_heat_gradient<2> (slae.solution, domain, "gradient-4.gpd");
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
    };

    void solve_nikola_elastic_problem (cst flag)
    {
        if (flag)
        {
            enum {x, y, z};
            Domain<2> domain;
            {
                // std::vector< dealii::Point< 2 > > v (6);
                //
                // v[0]  = dealii::Point<2>(0.0, -0.5);
                // v[1]  = dealii::Point<2>(1.0, -0.5);
                // v[2]  = dealii::Point<2>(0.0, 0.0);
                // v[3]  = dealii::Point<2>(1.0, 0.0);
                // v[4]  = dealii::Point<2>(0.0, 0.5);
                // v[5]  = dealii::Point<2>(1.0, 0.5);
                //
                // std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());
                // {
                //     dealii::CellData<2> tmp;
                //     tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.vertices[2]=3;tmp.vertices[3]=2;tmp.material_id=0;
                //     c .push_back (tmp);
                //     tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.vertices[2]=4;tmp.vertices[3]=2;tmp.material_id=0;
                //     c .push_back (tmp);
                // };
                // // c .push_back (dealii::CellData<2>{{0, 1, 3, 2}, 0});
                // // c .push_back (dealii::CellData<2>{{3, 5, 4, 2}, 1});
                //
                // dealii::SubCellData b;
                // {
                //     dealii::CellData<1> tmp;
                //     tmp.vertices[0]=4;tmp.vertices[1]=2;tmp.boundary_id=0;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=2;tmp.vertices[1]=0;tmp.boundary_id=0;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=0;tmp.vertices[1]=1;tmp.boundary_id=1;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=1;tmp.vertices[1]=3;tmp.boundary_id=2;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=3;tmp.vertices[1]=5;tmp.boundary_id=2;
                //     b.boundary_lines .push_back (tmp);
                //     tmp.vertices[0]=5;tmp.vertices[1]=4;tmp.boundary_id=3;
                //     b.boundary_lines .push_back (tmp);
                // };
                // // b.boundary_lines .push_back (dealii::CellData<1>{4, 2, 0});
                // // b.boundary_lines .push_back (dealii::CellData<1>{2, 0, 0});
                // // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
                // // b.boundary_lines .push_back (dealii::CellData<1>{1, 3, 2});
                // // b.boundary_lines .push_back (dealii::CellData<1>{3, 5, 2});
                // // b.boundary_lines .push_back (dealii::CellData<1>{5, 4, 3});
                //
                // dealii::GridReordering<2> ::reorder_cells (c);
                // domain.grid .create_triangulation_compatibility (v, c, b);
                //
                // domain.grid .refine_global (4);

                // set_quadrate<2> (domain.grid, 32.0, 96.0, 4);
                //
                //
                // vec<prmt::Point<2>> inner_border;
                // vec<prmt::Point<2>> outer_border;
                // // give_circ(inner_border, 4, 0.5, prmt::Point<2>(0.0, 0.0));
                // give_circ(outer_border, 4, 0.05, prmt::Point<2>(0.0, 0.0));
                // give_rectangle(inner_border, 1,
                //         prmt::Point<2>(-0.01, -0.01), prmt::Point<2>(0.01, 0.01));
                // // give_rectangle(outer_border, 1,
                // //         prmt::Point<2>(-1.0, -1.0), prmt::Point<2>(1.0, 1.0));
                // // give_rectangle(inner_border, 1,
                // //         prmt::Point<2>(0.25, -0.25), prmt::Point<2>(0.75, 0.25));
                // // give_rectangle(outer_border, 1,
                // //         prmt::Point<2>(0.0, -0.5), prmt::Point<2>(1.0, 0.5));
                // set_grid(domain.grid, outer_border, inner_border);

                set_tube(domain.grid, str("test_3.msh"), dealii::Point<2>(0.0,0.0), 1.0, 2.0, 1);

                // set_tube(domain.grid, str("circle_R2.msh"), dealii::Point<2>(0.0,0.0), 1.0, 2.0, 2);
                // set_tube(domain.grid, str("quadro_R2_R1.msh"), dealii::Point<2>(0.0,0.0), 1.0, 2.0, 1);
                // set_tube(domain.grid, str("quadro_dirka_R2_R1.msh"), dealii::Point<2>(0.0,0.0), 1.0, 2.0, 4);
                // set_crazy_tube(domain.grid, dealii::Point<2>(0.0,0.0), 1.0, 2.0, 6);
                // set_quadrate<2>(domain.grid, 
                //         -2.0, -1.0, 1.0, 2.0,
                //         -2.0, -1.0, 1.0, 2.0,
                //         4);


                // set_quadrate<2> (domain.grid, 
                //         // 0.0, 1.0/3.0, 2.0/3.0, 1.0, 
                //         -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
                //         -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
                //         5);
            };
            dealii::FESystem<2,2> fe 
                (dealii::FE_Q<2,2>(1), 2);
            domain.dof_init (fe);

            SystemsLinearAlgebraicEquations slae;
            ATools ::trivial_prepare_system_equations (slae, domain);

            LaplacianVector<2> element_matrix (domain.dof_handler.get_fe());
            element_matrix.C .resize (3);
            EPTools ::set_isotropic_elascity{yung : 100.0, puasson : 0.1}(element_matrix.C[1]);
            EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
            EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[2]);

            // T2.2
            vec<arr<arr<typename Nikola::SourceVector<2>::Func, 2>, 2>> U(3);
            U[0][x][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[0][x][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[0][y][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[0][y][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[1][x][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[1][x][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[1][y][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[1][y][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[2][x][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[2][x][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[2][y][x] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            U[2][y][y] = [&element_matrix] (const dealii::Point<2> &p) {return element_matrix.C[0][x][x][z][z];}; //Uz
            vec<arr<typename Nikola::SourceVector<2>::Func, 2>> tau(3);
            tau[0][x] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[0][y] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[1][x] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[1][y] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[2][x] = [] (const dealii::Point<2> &p) {return 0.0;};
            tau[2][y] = [] (const dealii::Point<2> &p) {return 0.0;};

            // Nikola::SourceVector<2> element_rhsv (U, tau, domain.dof_handler.get_fe());
            // SourceVector<2> element_rhsv (func, domain.dof_handler.get_fe());
            SourceVectorFeature<2> element_rhsv1 (U, domain.dof_handler.get_fe());
            SourceVectorPolyMaterials<2> element_rhsv2 (tau, domain.dof_handler.get_fe());

            Assembler ::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
            puts("scsdf");
            // Assembler ::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);
            Assembler ::assemble_rhsv<2> (slae.rhsv, element_rhsv1, domain.dof_handler);
            Assembler ::assemble_rhsv<2> (slae.rhsv, element_rhsv2, domain.dof_handler);

            dealii::SolverControl solver_control (10000, 1e-12);
            dealii::SolverCG<> solver (solver_control);
            solver.solve (
                    slae.matrix,
                    slae.solution,
                    slae.rhsv
                    ,dealii::PreconditionIdentity()
                    );


            dbl integral_x = 0.0;
            dbl integral_y = 0.0;
            dealii::Vector<dbl> s_values(slae.solution.size());
            s_values = 0.0;
            {
                dealii::QGauss<2>  quadrature_formula(2);

                dealii::FEValues<2> fe_values (domain.dof_handler.get_fe(), quadrature_formula,
                        dealii::update_quadrature_points | dealii::update_JxW_values |
                        dealii::update_values);

                cst dofs_per_cell = fe.dofs_per_cell;
                cst n_q_points = quadrature_formula.size();
                dealii::Vector<dbl> cell_value (dofs_per_cell);
                vec<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);


                auto cell = domain.dof_handler.begin_active();
                auto endc = domain.dof_handler.end();
                for (; cell != endc; ++cell)
                {
                    fe_values .reinit (cell);
                    cell_value = 0.0;

                    for (st i = 0; i < dofs_per_cell; ++i)
                        for (st q_point = 0; q_point < n_q_points; ++q_point)
                            cell_value[i] += fe_values.shape_value (i, q_point) *
                                fe_values.JxW(q_point);

                    cell->get_dof_indices (local_dof_indices);
                    for (st i = 0; i < dofs_per_cell; ++i)
                        s_values(local_dof_indices[i]) += cell_value(i);
                };
            };

            for (st i = 0; i < slae.solution.size(); ++i)
            {
                integral_x += !(i % 2) ? slae.solution(i) * s_values(i) : 0.0;
                integral_y += (i % 2) ? slae.solution(i) * s_values(i) : 0.0;
            };
            // printf("Integral %.10f %.10f\n", integral_x, integral_y);

            // for (st i = 0; i < slae.solution.size(); ++i)
            // {
            //     // slae.solution(i) -= (i % 2) ? integral_x : integral_y;
            // };
            // // // EPTools ::print_move<2> (indexes, domain.dof_handler, "move.gpd");
            EPTools ::print_move<2> (slae.solution, domain.dof_handler, "move-2.gpd");

            cdbl R_1 = 1.0;
            cdbl R_2 = 2.0;
            cdbl lambda_1 = element_matrix.C[1][x][x][y][y];
            cdbl lambda_2 = element_matrix.C[0][x][x][y][y];
            cdbl mu_1 = element_matrix.C[1][x][y][x][y];
            cdbl mu_2 = element_matrix.C[0][x][y][x][y];
            cdbl a_1 = (lambda_2 + mu_2);
            cdbl b_1 = - mu_2 / (R_2 * R_2);
            cdbl d_1 = - lambda_2 / 2.0;
            cdbl a_2 = (lambda_2 + mu_2) - (lambda_1 + mu_1);
            cdbl b_2 = - (mu_2 + lambda_1 + mu_1) / (R_1 * R_1);
            cdbl d_2 = (lambda_1 - lambda_2) / 2.0;
            cdbl C2 = (d_2 - a_2 / a_1 * d_1) / (b_2 - a_2 / a_1 * b_1);
            cdbl C1 = (d_1 - b_1 * C2) / a_1;
            printf("%f %f %f %f %f %f\n", a_1, b_1, d_1, a_2, b_2, d_2);
            printf("%f %f\n", (d_2 - a_2 / a_1 * d_1), (b_2 - a_2 / a_1 * b_1));
            printf("%f %f C1=%f C2=%f\n", lambda_1, mu_1, C1, C2);
            printf("%f\n", (lambda_2+2.0*mu_2)*(C1-C2/(R_2*R_2))+lambda_2*(C1+C2/(R_2*R_2))+lambda_2);
            printf("%f\n", (lambda_2+2.0*mu_2)*(C1-C2/(R_1*R_1))+lambda_2*(C1+C2/(R_1*R_1))+lambda_2);
            printf("%f\n", C1 * R_2 + C2 / R_2);

            dealii::Vector<dbl> anal(slae.solution.size());
            dealii::Vector<dbl> diff(slae.solution.size());
            dealii::Vector<dbl> digit(slae.solution.size());
            for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
            {
                for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
                {
                    dbl indx_x = cell->vertex_dof_index(i, x);
                    dbl indx_y = cell->vertex_dof_index(i, y);
                    auto p = cell->vertex(i);
                    dbl r = std::pow(p(x)*p(x) + p(y)*p(y), 0.5);
                    anal(indx_x) = (r < R_1) ? (C1+C2/(R_1*R_1))*r : C1*r+C2/r;
                    digit(indx_x) = std::abs(r) > 1e-5 ? 
                        // slae.solution(indx_x) * p(0) / r + slae.solution(indx_y) * p(1) / r :
                        -std::sqrt(std::pow(slae.solution(indx_x), 2.0) + std::pow(slae.solution(indx_y), 2.0)) :
                        0.0;
                    if (p(0) == -0.5)
                        printf("%f %f %f %f %f %f\n", digit(indx_x),
                                slae.solution(indx_x),
                                p(0) / r,
                                slae.solution(indx_y),
                                p(1) / r,
                                r);
                    digit(indx_y) = C1 * r;
                    diff(indx_x) = 
                        std::abs(anal(indx_x) - digit(indx_x));
                };
            };
            EPTools ::print_move<2> (digit, domain.dof_handler, "digit.gpd");
            EPTools ::print_move<2> (anal, domain.dof_handler, "anal.gpd");
        EPTools ::print_move<2> (diff, domain.dof_handler, "diff.gpd");
        // EPTools ::print_move<2> (slae.rhsv, domain.dof_handler, "rhsv.gpd");


    };
};

void solve_heat_conduction_problem_3d (cst flag)
{
    if (flag)
    {       
        enum {x, y, z};
        Domain<3> domain;
        {
            // dealii::GridGenerator::hyper_cube(domain.grid, 0.0, 2.0);
            // domain.grid.refine_global(1);
            set_cylinder(domain.grid, 0.344827, x, 4);
            // set_ball(domain.grid, 0.2927, 4);
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
            element_matrix.C[0][x][z] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            element_matrix.C[0][y][z] = 0.0;
            element_matrix.C[0][z][x] = 0.0;
            element_matrix.C[0][z][y] = 0.0;
            element_matrix.C[0][z][z] = 1.0;
            
            element_matrix.C[1][x][x] = 1.0;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][x][z] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 1.0;
            element_matrix.C[1][y][z] = 0.0;
            element_matrix.C[1][z][x] = 0.0;
            element_matrix.C[1][z][y] = 0.0;
            element_matrix.C[1][z][z] = 1.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
        };

        // auto func = [] (dealii::Point<3>) {return 0.0;};
        // SourceScalar<3> element_rhsv (func, domain.dof_handler.get_fe());
        vec<arr<typename SourceScalarFeature<3>::Func, 3>> U(2);
        // U[0][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;}; //Ux
        // U[0][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;}; //Uy
        // U[1][x] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;};
        // U[1][y] = [mu, nu, c0] (const dealii::Point<2> &p) {return 0.0;};
        U[0][x] = [] (const dealii::Point<3> &p) {return 0.0;}; //Ux
        U[0][y] = [] (const dealii::Point<3> &p) {return 0.0;}; //Uy
        U[0][z] = [] (const dealii::Point<3> &p) {return 1.0;}; //Uy
        U[1][x] = [] (const dealii::Point<3> &p) {return 0.0;};
        U[1][y] = [] (const dealii::Point<3> &p) {return 0.0;};
        U[1][z] = [] (const dealii::Point<3> &p) {return 1.0;};
        vec<typename SourceScalarPolyMaterials<3>::Func> tau(2);
        tau[0] = [] (const dealii::Point<3> &p) {return 0.0;};
        tau[1] = [] (const dealii::Point<3> &p) {return 0.0;};
        SourceScalarFeature<3> element_rhsv1 (U, domain.dof_handler.get_fe());
        SourceScalarPolyMaterials<3> element_rhsv2 (tau, domain.dof_handler.get_fe());

        Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler);
        // Assembler::assemble_rhsv<3> (slae.rhsv, element_rhsv, domain.dof_handler);
        Assembler::assemble_rhsv<3> (slae.rhsv, element_rhsv1, domain.dof_handler);
        Assembler::assemble_rhsv<3> (slae.rhsv, element_rhsv2, domain.dof_handler);

        vec<BoundaryValueScalar<3>> bound (1);
        bound[0].function      = [] (const dealii::Point<3> &p) {return p(1);};
        bound[0].boundary_id   = 0;
        bound[0].boundary_type = TBV::Dirichlet;

        // for (auto b : bound)
        //     ATools ::apply_boundary_value_scalar<3> (b) .to_slae (slae, domain);

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
        // HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "temperature.gpd", dealii::DataOutBase::gnuplot);
        // HCPTools ::print_temperature_slice (slae.solution, domain.dof_handler, "temperature_slice.gpd", y, 0.5);
        HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "temperature.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_heat_conductions<2> (
        //         slae.solution, element_matrix.C, domain, "heat_conductions");
        // HCPTools ::print_heat_gradient<2> (
        //         slae.solution, element_matrix.C, domain, "heat_gradient");

    };
};

void normalization(/* const dealii::DoFHandler<dim> &dof_h, */ dealii::Vector<dbl> &solution)
{
    dbl sum = 0.0;
    for (st i = 0; i < solution.size(); ++i)
    {
        sum += solution(i);
    };
    sum /= solution.size();
    for (st i = 0; i < solution.size(); ++i)
    {
        solution(i) -= sum;
    };
    // auto cell = dof_h.begin_active();
    // auto endc = dof_h.end();
    // for (; cell != endc; ++cell)
    // {
    //     for (st i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
    //     {
    // FOR (q_point, 0, num_quad_points)
    //     FOR (a, 0, dim)
    //         FOR (b, 0, dim)
    //         {
    //             // printf("%ld %ld %d\n", i, j, material_id);
    //             res += C[material_id][a][b] * 
    //                 fe_values.shape_grad (i, q_point)[a] *
    //                 fe_values.shape_grad (j, q_point)[b] *
    //                 fe_values.JxW(q_point);
    //         };
    //     };
    // };
};

template <st dim>
void get_micro_heat_flow (
        const Domain<dim> &domain, 
        const arr<dealii::Vector<dbl>, dim> &T,
        const vec<ATools::SecondOrderTensor> &C,
        arr<arr<dealii::Vector<dbl>, dim>, dim> &q)
{
    enum {x, y, z};
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            q[i][j] .reinit (domain.dof_handler.n_dofs());
            for (st k = 0; k < q[i][j].size(); ++k)
            {
                q[i][j](k) = 0.0;
            };
        };
    };
    vec<st> N(domain.dof_handler.n_dofs());
    for (st i = 0; i < N.size(); ++i)
    {
        N[i] = 0;
    };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            cst m_id = cell->material_id();
            for (st i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
            {
                const dealii::Point<dim> p = cell -> center(); //vertex(i);
                cst indx = cell ->vertex_dof_index (i, 0);
                for (st j = 0; j < 3; ++j)
                {
                    // dealii::Tensor<1, 2, dbl> grad =
                    auto grad =
                        dealii::VectorTools::point_gradient(domain.dof_handler, T[j], p);
                    for (st k = 0; k < 3; ++k)
                    {
                        // q[j][k](indx) = 0.0;//C[m_id][j][k];
                        for (st l = 0; l < 3; ++l)
                        {
                            q[j][k](indx) += C[m_id][k][l] * grad[l];
                            // q[j][k](indx) = grad[k];
                        };
                        q[j][k](indx) += C[m_id][j][k];
                        // q[j][k](indx) = C[m_id][j][k];
                            // q[j][k](indx) += c[m_id][j][k] * (grad[k] + 1.0);
                            // q[j][k](indx) += grad[k];
                            // q[j][k](indx) += C[m_id][j][k] * grad[k];
                            ++N[indx];
                    };
                };
            };
        };
    };
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            for (st k = 0; k < q[i][j].size(); ++k)
            {
                q[i][j](k) /= N[k];
            };
        };
    };
};

void solve_heat_conduction_problem_on_cell_3d (cst flag)
{
    if (flag)
    {  
        enum {x, y, z};
        // FILE *F;
        // F = fopen("square.gpd", "w");
        dbl size = 0.05;
        cdbl R = 0.25;
        cst n_p = 32;
        {
            Domain<3> domain;
            {
                // set_cylinder(domain.grid, 0.25, z, 3);
                // set_ball(domain.grid, 0.40057, 7);
                // set_ball(domain.grid, 0.4742, 4);
                set_cylinder_true(domain.grid, R, z, n_p, 5);
            };
            dealii::FE_Q<3> fe(1);
            domain.dof_init (fe);

            OnCell::SystemsLinearAlgebraicEquations<3> slae;
            OnCell::BlackOnWhiteSubstituter bows;
            // BlackOnWhiteSubstituter bows;

            LaplacianScalar<3> element_matrix (domain.dof_handler.get_fe());
            // {
            element_matrix.C .resize(2);
            element_matrix.C[1][x][x] = 1.0e1;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = 1.0e1;
            element_matrix.C[1][z][z] = 1.0e1;
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;
            element_matrix.C[0][z][z] = 1.0;
            // HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
            // };
            const bool scalar_type = 0;
            // OnCell::prepare_system_equations<scalar_type> (slae, bows, domain);
            // OnCell::prepare_system_equations_with_cubic_grid<3, 1> (slae, bows, domain);
            OnCell::prepare_system_equations_alternate<3, 1, 3> (slae, bows, domain);

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

                dealii::SolverControl solver_control (5000000, 1e-12);
                dealii::SolverCG<> solver (solver_control);
                solver.solve (
                        slae.matrix,
                        slae.solution[i],
                        slae.rhsv[i]
                        ,dealii::PreconditionIdentity()
                        );

                FOR(j, 0, slae.solution[i].size())
                    slae.solution[i][j] = slae.solution[i][bows.subst (j)];

                normalization(/* domain.dof_handler, */ slae.solution[i]);
            };

            arr<arr<dealii::Vector<dbl>, 3>, 3> heat_flow_reale;
            // OnCell::calculate_heat_flow<3>(slae.solution, slae.rhsv, element_matrix.C, heat_flow_reale);
            get_micro_heat_flow<3>(domain, slae.solution, element_matrix.C, heat_flow_reale);

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
            HCPTools ::print_temperature_slice (slae.solution[x], domain.dof_handler, "temperature_slice_x.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[x][x], domain.dof_handler, "heat_flow_slice_xx.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[y][y], domain.dof_handler, "heat_flow_slice_yy.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[z][z], domain.dof_handler, "heat_flow_slice_zz.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[x][y], domain.dof_handler, "heat_flow_slice_xy.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[z][y], domain.dof_handler, "heat_flow_slice_zy.gpd", z, 0.5);
            HCPTools ::print_temperature_slice (heat_flow_reale[x][z], domain.dof_handler, "heat_flow_slice_xz.gpd", z, 0.5);

            auto meta_coef = OnCell::calculate_meta_coefficients_scalar<3> (
                    domain.dof_handler, slae.solution, slae.rhsv, element_matrix.C);
            printf("META %.15f %.15f %.15f\n", meta_coef[x][x], meta_coef[y][y], meta_coef[x][y]);
            printf("META %.15f %.15f %.15f %.15f %.15f %.15f\n", 
                    meta_coef[x][x], meta_coef[y][y], meta_coef[z][z],
                    meta_coef[x][y], meta_coef[x][z], meta_coef[y][z]);
        };

    };
};

void print_move_slice_with_vera_func (const dealii::Vector<dbl> &move, 
        const dealii::DoFHandler<3> &dof_handler,
        const str file_name,
        cst ort,
        cdbl slice_coor,
        cdbl Ri,
        cdbl Ro,
        cdbl E,
        cdbl Nu,
        cdbl P)
{
    cdbl C1 = (1-Nu)/E*(P*Ri*Ri/(Ro*Ro-Ri*Ri));
    cdbl C2 = (1+Nu)/E*(P*Ri*Ri*Ro*Ro/(Ro*Ro-Ri*Ri));

    FILE* f_out;
    f_out = fopen (file_name.c_str(), "w");
    for (auto cell = dof_handler.begin_active (); cell != dof_handler.end (); ++cell)
    {
        for (st i = 0; i < dealii::GeometryInfo<3>::vertices_per_cell; ++i)
        {
            if (std::abs(cell->vertex(i)(ort) - slice_coor) < 1e-10)
            {
                cdbl r = std::sqrt(cell->vertex(i)(0)*cell->vertex(i)(0)+cell->vertex(i)(1)*cell->vertex(i)(1));
                fprintf(f_out, "%f %f %f %f %f %f %f %f\n",
                        cell->vertex(i)(0),
                        cell->vertex(i)(1),
                        cell->vertex(i)(2),
                        move(cell->vertex_dof_index(i, 0)),
                        move(cell->vertex_dof_index(i, 1)),
                        move(cell->vertex_dof_index(i, 2)),
                        C1*r+C2/r,
                        std::abs(
                        std::sqrt(
                        move(cell->vertex_dof_index(i, 0))*move(cell->vertex_dof_index(i, 0))+
                        move(cell->vertex_dof_index(i, 1))*move(cell->vertex_dof_index(i, 1))
                        ) -
                        (C1*r+C2/r))
                       );
            };
        };
    };
    fclose(f_out);
};

void solve_elastic_problem_3d (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        cdbl len_rod = 1.0;
        Domain<3> domain;
        // {
        //     dealii::GridGenerator::hyper_cube(domain.grid, 0.0, 1.0);
        //     // domain.grid.refine_global(2);
        //     // set_cylinder(domain.grid, 0.475, z, 5);
        //     // set_ball(domain.grid, 0.4, 3);
        //     // set_long_rod(domain.grid, len_rod, 0.4, 3);
        //     // domain.grid.refine_global(2);
        //     //
        //     // set_speciment(domain.grid, len_rod, 1.0, 1.0, 0.4 / 8.0, 1.0 / 8.0, arr<st,6>({2,2,2,2,2,2}), 5);
        //     domain.grid.begin_active()->face(0)->set_boundary_indicator(1);
        //     domain.grid.begin_active()->face(1)->set_boundary_indicator(2);
        //     domain.grid.begin_active()->face(2)->set_boundary_indicator(0);
        //     domain.grid.begin_active()->face(3)->set_boundary_indicator(0);
        //     domain.grid.begin_active()->face(4)->set_boundary_indicator(0);
        //     domain.grid.begin_active()->face(5)->set_boundary_indicator(0);
        //     domain.grid.refine_global(3);
        //
        //     cdbl radius = 0.1;
        //     cdbl radius_2 = 0.12;
        //     dealii::Point<2> center (0.5, 0.5);
        //     {
        //         // dealii::Point<2> center (0.5, 0.5);
        //         dealii::Triangulation<3>::active_cell_iterator
        //             cell = domain.grid .begin_active(),
        //                  end_cell = domain.grid .end();
        //         for (; cell != end_cell; ++cell)
        //         {
        //             dealii::Point<2> midle_p(0.0, 0.0);
        //
        //             for (size_t i = 0; i < 8; ++i)
        //             {
        //                 midle_p(0) += cell->vertex(i)(0);
        //                 midle_p(1) += cell->vertex(i)(1);
        //             };
        //             midle_p(0) /= 8.0;
        //             midle_p(1) /= 8.0;
        //
        //             // printf("%f %f\n", midle_p(0), midle_p(1));
        //
        //             if (center.distance(midle_p) < radius_2)
        //             {
        //                 cell->set_material_id(2);
        //                 //                puts("adf");
        //             }
        //             else
        //                 cell->set_material_id(0);
        //         };
        //     };
        //     {
        //         // dealii::Point<2> center (0.5, 0.5);
        //         dealii::Triangulation<3>::active_cell_iterator
        //             cell = domain.grid .begin_active(),
        //                  end_cell = domain.grid .end();
        //         for (; cell != end_cell; ++cell)
        //         {
        //             dealii::Point<2> midle_p(0.0, 0.0);
        //
        //             for (size_t i = 0; i < 8; ++i)
        //             {
        //                 midle_p(0) += cell->vertex(i)(0);
        //                 midle_p(1) += cell->vertex(i)(1);
        //             };
        //             midle_p(0) /= 8.0;
        //             midle_p(1) /= 8.0;
        //
        //             // printf("%f %f\n", midle_p(0), midle_p(1));
        //
        //             if (center.distance(midle_p) < radius)
        //             {
        //                 cell->set_material_id(1);
        //                 //                puts("adf");
        //             }
        //             // else
        //             //     cell->set_material_id(0);
        //         };
        //     };
        // };
        cdbl Ri = 0.5;
        cdbl Ro = 11.0;
        vec<dealii::Point<2>> radius_vector;
        vec<dealii::Point<2>> radius_vector_cell;
        {
            auto center = dealii::Point<2>(0.0, 0.0);
            dealii::GridGenerator::cylinder_shell(domain.grid, 1.0, Ri, Ro, 64, 1);
            // domain.grid.refine_global(4);
            // domain.grid.begin_active()->face(0)->set_boundary_indicator(1);
            // domain.grid.begin_active()->face(1)->set_boundary_indicator(1);
            // bool flag = false;
            auto cell = domain.grid .begin_active();
            auto end_cell = domain.grid .end();
            for (; cell != end_cell; ++cell)
            {
                for (st i = 0; i < 6; ++i)
                {
                    if (cell->face(i)->at_boundary())
                    {
                        auto p = cell->face(i)->center();
                        auto p2d = dealii::Point<2>(p(0), p(1));
                        // printf("%f %f\n", std::abs(center.distance(p2d) - Ri), center.distance(p2d));
                        if (std::abs(center.distance(p2d) - Ri) < 0.1)
                        {
                            radius_vector .push_back(p2d);
                            cell->face(i)->set_boundary_indicator(radius_vector.size()+2);
                            // printf("dfgdfgvfvbf\n");
                        };
                        // if (!flag)
                        if (std::abs(p(2) - 0.0) < 1.0e-5)
                        {
                            cell->face(i)->set_boundary_indicator(1);
                            // flag = true;
                        };
                        if (std::abs(p(2) - 1.0) < 1.0e-5)
                        {
                            cell->face(i)->set_boundary_indicator(2);
                        };
                    };
                };

                dealii::Point<2> midle_p(0.0, 0.0);
                for (size_t i = 0; i < 8; ++i)
                {
                    midle_p(0) += cell->vertex(i)(0);
                    midle_p(1) += cell->vertex(i)(1);
                };
                midle_p(0) /= 8.0;
                midle_p(1) /= 8.0;
                cell->set_material_id(radius_vector_cell.size());
                radius_vector_cell .push_back(midle_p);
            };
            domain.grid.refine_global(2);
        };
        dealii::FESystem<3,3> fe 
            (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        // element_matrix.C .resize (3);
        // // EPTools ::set_isotropic_elascity{yung : 100.0, puasson : 0.15}(element_matrix.C[0]);
        // // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);
        // // EPTools ::set_isotropic_elascity{yung : 0.0, puasson : 0.0}(element_matrix.C[1]);
        // // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[1]);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[2]);

        ATools::FourthOrderTensor C;
        {
            ATools::FourthOrderTensor Cxy;
            std::ifstream in ("meta_coef.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &Cxy, sizeof Cxy);
            in.close ();
            arr<arr<dbl,3>,3> A = {
                arr<dbl,3>{1.0, 0.0, 0.0},
                arr<dbl,3>{0.0,  0.0, -1.0},
                arr<dbl,3>{0.0,  1.0, 0.0}
            };
            // arr<arr<dbl,3>,3> A = {
            //     arr<dbl,3>{1.0, 0.0, 0.0},
            //     arr<dbl,3>{0.0,  1.0, 0.0},
            //     arr<dbl,3>{0.0,  0.0, 1.0}
            // };

            // for (size_t i = 0; i < 9; ++i)
            // {
            //     uint8_t im = i / (3 + 1);
            //     uint8_t in = i % (3 + 1);
            //
            //     for (size_t j = 0; j < 9; ++j)
            //     {
            //         uint8_t jm = j / (3 + 1);
            //         uint8_t jn = j % (3 + 1);
            //
            //         if (std::abs(Cxy[im][in][jm][jn]) > 0.0000001)
            //             printf("\x1B[31m%f\x1B[0m   ", 
            //                     Cxy[im][in][jm][jn]);
            //         else
            //             printf("%f   ", 
            //                     Cxy[im][in][jm][jn]);
            //     };
            //     for (size_t i = 0; i < 2; ++i)
            //         printf("\n");
            // };
            //     {
            //     auto newcoef = unphysical_to_physicaly (Cxy);
            // printf("%f %f %f %f %f %f %f %f %f %f %f\n",
            //             newcoef[0][0][0][0],
            //             newcoef[0][0][1][1],
            //             newcoef[0][0][2][2],
            //             newcoef[1][1][0][0],
            //             newcoef[1][1][1][1],
            //             newcoef[1][1][2][2],
            //             newcoef[2][2][0][0],
            //             newcoef[2][2][1][1],
            //             newcoef[2][2][2][2],
            //             Cxy[0][1][0][1],
            //             Cxy[0][2][0][2]
            //             );
            //     };

            for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
            for (st k = 0; k < 3; ++k)
            for (st l = 0; l < 3; ++l)
            {
                C[i][j][k][l] = 0.0;

                for (st a = 0; a < 3; ++a)
                for (st b = 0; b < 3; ++b)
                for (st c = 0; c < 3; ++c)
                for (st d = 0; d < 3; ++d)
                {
                    C[i][j][k][l] += 
                        // C[i][j][k][l];
                        A[i][a]*A[j][b]*A[k][c]*A[l][d]*Cxy[a][b][c][d];

                };
            };


            // for (size_t i = 0; i < 9; ++i)
            // {
            //     uint8_t im = i / (3 + 1);
            //     uint8_t in = i % (3 + 1);
            //
            //     for (size_t j = 0; j < 9; ++j)
            //     {
            //         uint8_t jm = j / (3 + 1);
            //         uint8_t jn = j % (3 + 1);
            //
            //         if (std::abs(C[im][in][jm][jn]) > 0.0000001)
            //             printf("\x1B[31m%f\x1B[0m   ", 
            //                     C[im][in][jm][jn]);
            //         else
            //             printf("%f   ", 
            //                     C[im][in][jm][jn]);
            //     };
            //     for (size_t i = 0; i < 2; ++i)
            //         printf("\n");
            // };
            //     {
            //     auto newcoef = unphysical_to_physicaly (C);
            // printf("%f %f %f %f %f %f %f %f %f %f %f\n",
            //             newcoef[0][0][0][0],
            //             newcoef[0][0][1][1],
            //             newcoef[0][0][2][2],
            //             newcoef[1][1][0][0],
            //             newcoef[1][1][1][1],
            //             newcoef[1][1][2][2],
            //             newcoef[2][2][0][0],
            //             newcoef[2][2][1][1],
            //             newcoef[2][2][2][2],
            //             C[0][1][0][1],
            //             C[0][2][0][2]
            //             );
            //     };

        };
            // exit(1);

        element_matrix.C .resize (radius_vector_cell.size());
        for (st n = 0; n < radius_vector_cell.size(); ++n)
        {
            cdbl hypo = sqrt(
                    radius_vector_cell[n](x)*radius_vector_cell[n](x)+
                    radius_vector_cell[n](y)*radius_vector_cell[n](y)
                    );
            cdbl sin = radius_vector_cell[n](y)/hypo;
            cdbl cos = radius_vector_cell[n](x)/hypo;
            arr<arr<dbl,3>,3> A = {
                arr<dbl,3>{cos, -sin, 0.0},
                arr<dbl,3>{sin,  cos, 0.0},
                arr<dbl,3>{0.0,  0.0, 1.0}
            };
            // arr<arr<dbl,3>,3> A = {
            //     arr<dbl,3>{1.0, 0.0, 0.0},
            //     arr<dbl,3>{0.0,  1.0, 0.0},
            //     arr<dbl,3>{0.0,  0.0, 1.0}
            // };

            // for (size_t i = 0; i < 9; ++i)
            // {
            //     uint8_t im = i / (3 + 1);
            //     uint8_t in = i % (3 + 1);
            //
            //     for (size_t j = 0; j < 9; ++j)
            //     {
            //         uint8_t jm = j / (3 + 1);
            //         uint8_t jn = j % (3 + 1);
            //
            //         if (std::abs(C[im][in][jm][jn]) > 0.0000001)
            //             printf("\x1B[31m%f\x1B[0m   ", 
            //                     C[im][in][jm][jn]);
            //         else
            //             printf("%f   ", 
            //                     C[im][in][jm][jn]);
            //     };
            //     for (size_t i = 0; i < 2; ++i)
            //         printf("\n");
            // };

            for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
            for (st k = 0; k < 3; ++k)
            for (st l = 0; l < 3; ++l)
            {
                element_matrix.C[n][i][j][k][l] = 0.0;

                for (st a = 0; a < 3; ++a)
                for (st b = 0; b < 3; ++b)
                for (st c = 0; c < 3; ++c)
                for (st d = 0; d < 3; ++d)
                {
                    element_matrix.C[n][i][j][k][l] += 
                        // C[i][j][k][l];
                        A[i][a]*A[j][b]*A[k][c]*A[l][d]*C[a][b][c][d];

                };
            };

            // printf("\n");
            // for (size_t i = 0; i < 9; ++i)
            // {
            //     uint8_t im = i / (3 + 1);
            //     uint8_t in = i % (3 + 1);
            //
            //     for (size_t j = 0; j < 9; ++j)
            //     {
            //         uint8_t jm = j / (3 + 1);
            //         uint8_t jn = j % (3 + 1);
            //
            //         if (std::abs(element_matrix.C[n][im][in][jm][jn]) > 0.0000001)
            //             printf("\x1B[31m%f\x1B[0m   ", 
            //                     element_matrix.C[n][im][in][jm][jn]);
            //         else
            //             printf("%f   ", 
            //                     element_matrix.C[n][im][in][jm][jn]);
            //     };
            //     for (size_t i = 0; i < 2; ++i)
            //         printf("\n");
            // };
            // break;
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[n]);
        };
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
            // exit(1);

        // const dbl abld = 
        //     element_matrix.C[0][x][x][x][x] +
        //     // element_matrix.C[0][x][x][x][y] +
        //     element_matrix.C[0][y][x][x][x];
        //     // element_matrix.C[0][y][x][x][y];
        // printf("AAAAAA %f\n", abld);
        // arr<std::function<dbl (const dealii::Point<3>&)>, 3> func {
        // // [=] (const dealii::Point<2>) {return -2.0*abld;},
        // [] (const dealii::Point<3>) {return 0.0;},
        // [] (const dealii::Point<3>) {return 0.0;},
        // [] (const dealii::Point<3>) {return 0.0;}
        // };
        // auto func = [] (dealii::Point<2>) {return arr<dbl, 2>{-2.0, 0.0};};
        arr<std::function<dbl (const dealii::Point<3>&)>, 3> func {
            [] (const dealii::Point<3>) {return 0.0;},
            [] (const dealii::Point<3>) {return 0.0;},
            [] (const dealii::Point<3>) {return 0.0;}
        };
        SourceVector<3> element_rhsv (func, domain.dof_handler.get_fe());

        // vec<arr<arr<typename Nikola::SourceVector<3>::Func, 3>, 3>> U(2);
        // U[0][x][x] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][x][y] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][x][z] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][y][x] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][y][y] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][y][z] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][z][x] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][z][y] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // U[0][z][z] = [&element_matrix] (const dealii::Point<3> &p) {return 0.0;}; 
        // vec<arr<typename Nikola::SourceVector<3>::Func, 3>> tau(2);
        // tau[0][x] = [] (const dealii::Point<3> &p) {return 0.0;};
        // tau[0][y] = [] (const dealii::Point<3> &p) {return 0.0;};
        // tau[0][z] = [] (const dealii::Point<3> &p) {return 0.0;};
        // tau[1][x] = [] (const dealii::Point<3> &p) {return 0.0;};
        // tau[1][y] = [] (const dealii::Point<3> &p) {return 0.0;};
        // tau[1][z] = [] (const dealii::Point<3> &p) {return 0.0;};
        //
        // SourceVectorFeature<3> element_rhsv1 (U, domain.dof_handler.get_fe());
        // SourceVectorPolyMaterials<3> element_rhsv2 (tau, domain.dof_handler.get_fe());
        //
        Assembler ::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler);
        // // Assembler ::assemble_rhsv<3> (slae.rhsv, element_rhsv, domain.dof_handler);
        // Assembler ::assemble_rhsv<3> (slae.rhsv, element_rhsv1, domain.dof_handler);
        // Assembler ::assemble_rhsv<3> (slae.rhsv, element_rhsv2, domain.dof_handler);

        // vec<BoundaryValueVector<3>> bound (2);
        // bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{-1.0, 0.0, 0.0};};
        // bound[0].boundary_id   = 1;
        // // bound[0].boundary_type = TBV::Dirichlet;
        // bound[0].boundary_type = TBV::Neumann;
        // bound[1].function      = [] (const dealii::Point<3> &p) {
        //     // return arr<dbl, 3>{0.0, 0.0, 0.0};};
        //     if (p(0) == 1.0)
        //         return arr<dbl, 3>{1.0, 0.0, 0.0};
        //     else if (p(0) == 0.0)
        //         return arr<dbl, 3>{-1.0, 0.0, 0.0};
        //     else
        //         return arr<dbl, 3>{0.0, 0.0, 0.0};};
        // bound[1].boundary_id   = 2;
        // // bound[1].boundary_type = TBV::Dirichlet;
        // bound[1].boundary_type = TBV::Neumann;

        // vec<BoundaryValueVector<3>> bound (2);
        // bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{-1.0, 0.0, 0.0};};
        // bound[0].boundary_id   = 1;
        // // bound[0].boundary_type = TBV::Neumann;
        // bound[0].boundary_type = TBV::Dirichlet;
        // bound[1].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{1.0, 0.0, 0.0};};
        // bound[1].boundary_id   = 2;

        // vec<BoundaryValueVector<3>> bound (2);
        // bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{1.0, 0.0, 0.0};};
        // bound[0].boundary_id   = 1;
        // // bound[0].boundary_type = TBV::Neumann;
        // bound[0].boundary_type = TBV::Dirichlet;
        // bound[1].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 1.0};};
        // bound[1].boundary_id   = 2;
        // // bound[1].boundary_type = TBV::Neumann;
        // bound[1].boundary_type = TBV::Dirichlet;

        cdbl P = 1.0;
        vec<BoundaryValueVector<3>> bound (radius_vector.size()+2);
        bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[0].boundary_id   = 1;
        bound[0].boundary_type = TBV::Neumann;
        bound[1].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[1].boundary_id   = 2;
        bound[1].boundary_type = TBV::Neumann;
        for (st i = 0; i < radius_vector.size(); ++i)
        {
            cdbl a = radius_vector[i](0);
            cdbl b = radius_vector[i](1);
            cdbl tmp = 1/std::sqrt(a*a + b*b);
            cdbl Px = P*a*tmp;
            cdbl Py = P*b*tmp;
            // printf("%f %f\n", Px, Py);
            bound[i+2].function      = [Px, Py] (const dealii::Point<3> &p) {return arr<dbl, 3>{Px, Py, 0.0};};
            bound[i+2].boundary_id   = i+3;
            bound[i+2].boundary_type = TBV::Neumann;
            // bound[i].boundary_type = TBV::Dirichlet;
        };
        // bound[bound.size()-1].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        // bound[bound.size()-1].boundary_id   = 1;
        // // bound[1].boundary_type = TBV::Neumann;
        // bound[bound.size()-1].boundary_type = TBV::Dirichlet;

        for (auto b : bound)
            ATools ::apply_boundary_value_vector<3> (b) .to_slae (slae, domain);

        dealii::SolverControl solver_control (1000000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );
        // EPTools ::print_move<3> (slae.solution, domain.dof_handler, "move.gpd");
        // EPTools ::print_move<3> (slae.solution, domain.dof_handler, "move.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "move.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_temperature_slice (slae.solution, domain.dof_handler, "temperature_slice.gpd", x, len_rod);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice_z.gpd", z, 0.5);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice_y.gpd", y, 0.0);

        // print_move_slice_with_vera_func (
        //         slae.solution, domain.dof_handler, "move_slice_with_vera_func.gpd", z, 0.0,
        //         Ri, Ro, 1.0, 0.25, P);
        // EPTools ::print_elastic_deformation_mean_other_3d_binary (
        //         slae.solution, domain.dof_handler, "deform_other.gpd");
        // dbl sum_x = 0.0;
        // dbl sum_y = 0.0;
        // st n_x = 0;
        // st n_y = 0;
        // for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
        // {
        //     for (st i = 0; i < dealii::GeometryInfo<3>::vertices_per_cell; ++i)
        //     {
        //         if (std::abs(cell->vertex(i)(x) - 1.0) < 1e-10)
        //         {
        //             sum_x += slae.solution(cell->vertex_dof_index(i, x)); 
        //             n_x++;
        //         };
        //         if (std::abs(cell->vertex(i)(y) - 1.0) < 1e-10)
        //         {
        //             sum_y += slae.solution(cell->vertex_dof_index(i, y)); 
        //             n_y++;
        //         };
        //     };
        // };
        // printf("%f %f %ld %ld %f %f E=%f nu=%f\n", 
        //         sum_x, sum_y, n_x, n_y, sum_x/n_x*2.0, sum_y/n_y*2.0, 0.5/(sum_x/n_x), (sum_y/n_y)/(sum_x/n_x));
    };
};

void solve_elastic_problem_on_cell_3d (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<3> domain;
        {
            set_cylinder(domain.grid, 0.4, z, 4);
            // set_ball(domain.grid, 0.4, 4);
                // set_rect_3d(domain.grid,
                //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
                //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
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

        arr<str, 6> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd", "move_xz.gpd", "move_yz.gpd"};
        for (st i = 0; i < 6; ++i)
        {
            // EPTools ::print_move<3> (slae.solution[i], domain.dof_handler, vr[i]);
            EPTools ::print_move_slice (slae.solution[i], domain.dof_handler, vr[i], z, 0.5);
        };
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice.gpd", z, 0.5);

        auto meta_coef = OnCell::calculate_meta_coefficients_3d_elastic<3> (
                domain.dof_handler, slae, element_matrix.C);

        {
        std::ofstream out ("meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

        {
        std::ofstream out ("solution_0.bin", std::ios::out | std::ios::binary);
        for (st i = 0; i < slae.solution[0].size(); ++i)
        {
        out.write ((char *) &(slae.solution[0][i]), sizeof slae.solution[0][0]);
        };
        out.close ();
        };

        {
        std::ofstream out ("solution_size.bin", std::ios::out | std::ios::binary);
        auto size = slae.solution[0].size();
        out.write ((char *) &size, sizeof size);
        out.close ();
        printf("SIZE %ld %ld %ld %f\n", size, slae.solution[0].size(), 8 * slae.solution[0].size(),
                slae.solution[0][10]);
        };

        // for (size_t i = 0; i < 9; ++i)
        // {
        //     uint8_t im = i / (dim + 1);
        //     uint8_t in = i % (dim + 1);
        //
        //     for (size_t j = 0; j < 9; ++j)
        //     {
        //         uint8_t jm = j / (dim + 1);
        //         uint8_t jn = j % (dim + 1);
        //
        //         if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
        //             printf("\x1B[31m%f\x1B[0m   ", 
        //                     meta_coef[im][in][jm][jn]);
        //         else
        //             printf("%f   ", 
        //                     meta_coef[im][in][jm][jn]);
        //     };
        //     for (size_t i = 0; i < 2; ++i)
        //         printf("\n");
        // };
        // // printf("Meta_xxxx %f\n", meta_coef[x][x][x][x]);
        // // printf("Meta_yxxy %f\n", meta_coef[y][x][x][y]);
        // // printf("Meta_yyxx %f\n", meta_coef[y][y][x][x]);
        // // printf("Meta_xzxz %f\n", meta_coef[x][z][x][z]);
        // // printf("Meta_zzzz %f\n", meta_coef[z][z][z][z]);
        // // printf("Meta_zzyy %f\n", meta_coef[z][z][y][y]);
        // // printf("Meta_zyzy %f\n", meta_coef[z][y][z][y]);
        // // print_tensor<6*6>(meta_coef);
        // {
        // auto newcoef = unphysical_to_physicaly (meta_coef);
        // // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         newcoef[0][0][0][0],
        //         newcoef[0][0][1][1],
        //         newcoef[0][0][2][2],
        //         newcoef[1][1][0][0],
        //         newcoef[1][1][1][1],
        //         newcoef[1][1][2][2],
        //         newcoef[2][2][0][0],
        //         newcoef[2][2][1][1],
        //         newcoef[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        //
        //         printf("\n");
        //         printf("\n");
        //
        // for (size_t i = 0; i < 9; ++i)
        // {
        //     uint8_t im = i / (dim + 1);
        //     uint8_t in = i % (dim + 1);
        //
        //     for (size_t j = 0; j < 9; ++j)
        //     {
        //         uint8_t jm = j / (dim + 1);
        //         uint8_t jn = j % (dim + 1);
        //
        //         if (std::abs(newcoef[im][in][jm][jn]) > 0.0000001)
        //             printf("\x1B[31m%f\x1B[0m   ", 
        //                     newcoef[im][in][jm][jn]);
        //         else
        //             printf("%f   ", 
        //                     newcoef[im][in][jm][jn]);
        //     };
        //     for (size_t i = 0; i < 2; ++i)
        //         printf("\n");
        // };
        // };

    };
};

void solve_approx_cell_heat_problem (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        // FILE *F;
        // F = fopen("square.gpd", "w");
        cdbl c = 0.78;
        cdbl lambda = 2.0e0;
        cdbl R_in = sqrt((c / dealii::numbers::PI));
        dbl size = sqrt(c);
        size = 0.5;
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
                    {0, 0, 1, 0},
                    {0, 1, 1, 0},
                    {1, 1, 1, 1},
                    {1, 1, 1, 1}
                };
                const double dot[5] = 
                {
                    (0.0),
                    (0.5 - 0.5 / 2.0),
                    (0.5),
                    (0.5 + 0.5 / 2.0),
                    (1.0)
                };
                // ::set_tria <5> (domain.grid, dot, material_id);
                // domain.grid .refine_global (1);
                set_rect(domain.grid,
                        dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 0.5 / 2.0)),
                        dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 0.5 / 2.0)), 3);
                // set_circ(domain.grid, R_in, 4); //0.344827, 2);
                // set_circ_in_hex(domain.grid, 0.3, 6);
                // ::set_hexagon_grid_pure (domain.grid, 1.0, 0.5);
            // set_tube(domain.grid, str("grid-igor_ss8.40749_h1_rho0.4.msh"), 
            //         dealii::Point<2>(4.0,4.0), 1.0, 2.0, 1);
                // domain.grid .refine_global (1);
                {
                    std::ofstream out ("grid-igor.eps");
                    dealii::GridOut grid_out;
                    grid_out.write_eps (domain.grid, out);
                };
            };
            dealii::FE_Q<2> fe(1);
            domain.dof_init (fe);

            OnCell::SystemsLinearAlgebraicEquations<1> slae;
            OnCell::BlackOnWhiteSubstituter bows;

            LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
            element_matrix.C .resize(2);
            element_matrix.C[1][x][x] = lambda;
            element_matrix.C[1][x][y] = 0.0;
            element_matrix.C[1][y][x] = 0.0;
            element_matrix.C[1][y][y] = lambda;
            element_matrix.C[0][x][x] = 1.0;
            element_matrix.C[0][x][y] = 0.0;
            element_matrix.C[0][y][x] = 0.0;
            element_matrix.C[0][y][y] = 1.0;

            const bool scalar_type = 0;
            OnCell::prepare_system_equations_with_cubic_grid<2, 1> (slae, bows, domain);

            OnCell::Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler, bows);

            cst number_of_approx = 5; // Начиная с нулевой
            arr<arr<i32, 3>, number_of_approx> approximations = {
                arr<i32, 3>{1, 0, 0},
                arr<i32, 3>{0, 1, 0},
                arr<i32, 3>{2, 0, 0},
                arr<i32, 3>{0, 2, 0},
                arr<i32, 3>{1, 1, 0}};
            OnCell::ArrayWithAccessToVector<dbl> meta_coefficient(number_of_approx);
            OnCell::ArrayWithAccessToVector<dealii::Vector<dbl>> psi_func (number_of_approx);
            OnCell::ArrayWithAccessToVector<dealii::Vector<dbl>> N_func (number_of_approx);
            for (auto &&a : meta_coefficient.content)
                for (auto &&b : a)
                    for (auto &&c : b)
                        c = 0.0;
            for (auto &&a : psi_func.content)
                for (auto &&b : a)
                    for (auto &&c : b)
                        c .reinit (slae.solution[0].size());
            psi_func[arr<i32, 3>{0, 0, 0}] = 1.0; // Нулевое приближение ячейковой функции равно 1.0
            for (auto &&a : N_func.content)
                for (auto &&b : a)
                    for (auto &&c : b)
                        c .reinit (slae.solution[0].size());

            dealii::Vector<dbl> v(0);
            // printf("%d\n", v == dealii::Vector<dbl>(0));
            auto mean_coefficient = 
                OnCell::calculate_mean_coefficients<2> (domain.dof_handler, element_matrix.C);
            auto area_of_domain = 
                OnCell::calculate_area_of_domain<2> (domain.dof_handler);
            // puts("123234234");
            // auto approximation = approximations[0];
            for (auto &&approximation : approximations)
            {
                slae.solution[0] = 0.0;
                slae.rhsv[0] = 0.0;
                printf("scsdcdfvdf %d\n", approximation);

                OnCell::SourceScalarApprox<2> element_rhsv (approximation,
                        element_matrix.C, 
                        meta_coefficient,
                        psi_func,
                        // &psi_func,
                        domain.dof_handler.get_fe());
                // printf("%d\n", slae.rhsv[0].size());
                OnCell::Assembler::assemble_rhsv<2> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);
                printf("Integ %f\n", element_rhsv.tmp);

                dealii::SolverControl solver_control (500000, 1e-12);
                dealii::SolverCG<> solver (solver_control);
                solver.solve (
                        slae.matrix,
                        slae.solution[0],
                        slae.rhsv[0]
                        ,dealii::PreconditionIdentity()
                        );
                FOR(i, 0, slae.solution[0].size())
                    slae.solution[0][i] = slae.solution[0][bows.subst (i)];

                psi_func[approximation] = slae.solution[0];
                N_func[approximation] = slae.rhsv[0];

                if ((approximation[0] == 1) and (approximation[1] == 0))
                {
                    st len_vector_solution = domain.dof_handler.n_dofs();
                    dbl mean_heat_flow;

                    mean_heat_flow = 0.0;
                    // for (st k = 0; k < len_vector_solution; ++k)
                    //     printf("%f\n", slae.rhsv[0](k));

                    for (st k = 0; k < len_vector_solution; ++k)
                        mean_heat_flow += psi_func[approximations[x]](k) * (-N_func[approximations[x]](k));

                    mean_heat_flow /= area_of_domain;

                    meta_coefficient[approximations[2]] = mean_coefficient[x][x] + mean_heat_flow;
                };

                if ((approximation[0] == 0) and (approximation[1] == 1))
                {
                    st len_vector_solution = domain.dof_handler.n_dofs();
                    dbl mean_heat_flow;

                    mean_heat_flow = 0.0;

                    for (st k = 0; k < len_vector_solution; ++k)
                        mean_heat_flow += psi_func[approximations[y]](k) * (-N_func[approximations[y]](k));


                    mean_heat_flow /= area_of_domain;

                    meta_coefficient[approximations[3]] = mean_coefficient[y][y] + mean_heat_flow;
                };

                if ((approximation[0] == 2) and (approximation[1] == 0))
                {
                    st len_vector_solution = domain.dof_handler.n_dofs();
                    dbl mean_heat_flow;

                    mean_heat_flow = 0.0;

                    for (st k = 0; k < len_vector_solution; ++k)
                        mean_heat_flow += psi_func[approximations[x]](k) * (-N_func[approximations[y]](k));


                    mean_heat_flow /= area_of_domain;

                    meta_coefficient[approximations[4]] = mean_coefficient[x][y] + mean_heat_flow;
                };

                printf("META %f %f %f\n", 
                        meta_coefficient[approximations[2]], 
                        meta_coefficient[approximations[3]], 
                        meta_coefficient[approximations[4]]);
            };

            {
            arr<str, number_of_approx> vr = {
                "temperature_approx_x.gpd", 
                "temperature_approx_y.gpd",
                "temperature_approx_2x.gpd",
                "temperature_approx_2y.gpd",
                "temperature_approx_xy.gpd"};
            FOR(i, 0, approximations.size())
                HCPTools ::print_temperature<2> (
                        psi_func[approximations[i]], 
                        domain.dof_handler, vr[i]);
            }; 
            // {
            // arr<str, 3> vr = {
            //     "rhsv_approx_x.gpd", 
            //     "rhsv_approx_y.gpd",
            //     "rhsv_approx_2x.gpd"};
            // FOR(i, 0, approximations.size())
            //     HCPTools ::print_temperature<2> (
            //             N_func[approximations[i]],
            //             domain.dof_handler, vr[i]);
            // };
            // {
            // arr<str, 3> vr = {
            //     "grad_approx_x.gpd", 
            //     "grad_approx_y.gpd",
            //     "grad_approx_2x.gpd"};
            // FOR(i, 0, approximations.size())
            //     HCPTools ::print_heat_gradient<2> (
            //             psi_func[approximations[i]], 
            //             domain, vr[i]);
            // }; 
            // {
            // arr<str, 3> vr = {
            //     "heat_conductions_approx_x.gpd", 
            //     "heat_conductions_approx_y.gpd",
            //     "heat_conductions_approx_2x.gpd"};
            // FOR(i, 0, approximations.size())
            //     HCPTools ::print_heat_conductions<2> (
            //             psi_func[approximations[i]], 
            //             element_matrix.C,
            //             domain, vr[i]);
            // }; 

            // auto meta_coef = OnCell::calculate_meta_coefficients_scalar<2> (
            //         domain.dof_handler, slae.solution, slae.rhsv, element_matrix.C);
            // printf("META %.15f %.15f %.15f\n", meta_coef[x][x], meta_coef[y][y], meta_coef[x][y]);

        };
            // fclose(F);
    };
};

void create_arbitrary_grid_with_circle_include (
        Domain<2> &domain,
        cdbl radius,
        cst num_border_point,
        cst num_includ_point)
{
    vec<prmt::Point<2>> border;
    vec<st> type_border;
    give_rectangle_with_border_condition(
            border,
            type_border,
            arr<st, 4>{1,3,2,4},
            num_border_point,
            prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
    vec<vec<prmt::Point<2>>> inclusion(1);
    // cdbl radius_2 = 0.255;
    dealii::Point<2> center (0.5, 0.5);
    give_circ(inclusion[0], num_includ_point, radius, prmt::Point<2>(center));
    // give_circ(inclusion[1], 40, radius_2, prmt::Point<2>(center));
    ::set_grid(domain.grid, border, inclusion, type_border);

    {
        std::ofstream out ("grid-igor.eps");
        dealii::GridOut grid_out;
        grid_out.write_eps (domain.grid, out);
    };

    {
        dealii::Triangulation<2>::active_cell_iterator
            cell = domain.grid .begin_active(),
                 end_cell = domain.grid .end();
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

            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
            }
            else
            {
                cell->set_material_id(0);
            };
        };
    };
};

void solve_cell_elastic_problem_and_print_along_line (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<2> domain;
        create_arbitrary_grid_with_circle_include (domain, 0.25, 1, 100);

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
        solve_heat_problem_on_cell_aka_torsion_rod<2> (
                domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);

        arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
        for (st i = 0; i < 4; ++i)
        {
            EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
        };
        // EPTools ::print_move<2> (problem_of_torsion_rod_slae.solution[0], domain.dof_handler, "move_xz.gpd");
        // EPTools ::print_elastic_stress (slae.solution[0], domain.dof_handler, 
        //         element_matrix.C[0], "cell_stress_xx.gpd");
        // EPTools ::print_elastic_deformation (slae.solution[0], domain.dof_handler, "cell_deform.gpd");
        // EPTools ::print_elastic_deformation_mean (slae.solution[0], domain.dof_handler, "cell_deform_mean.gpd");
    };
};

void solve_approx_cell_elastic_problem (cst flag)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<3> domain;
        {
            set_cylinder(domain.grid, 0.25, y, 3);
            // set_ball(domain.grid, 0.4, 3);
            // set_rect_3d(domain.grid,
            //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
            //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
        };
        dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        OnCell::SystemsLinearAlgebraicEquations<1> slae;
        OnCell::BlackOnWhiteSubstituter bows;

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.2}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.28}(element_matrix.C[1]);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
        EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(element_matrix.C[1]);

        OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);

        OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

        cst number_of_approx = 2; // Начиная с нулевой
        // arr<arr<i32, 3>, number_of_approx> approximations = {
        //     arr<i32, 3>({1, 0, 0}),
        //     arr<i32, 3>({0, 1, 0})};
            // arr<i32, 3>{2, 0, 0}};
        OnCell::ArrayWithAccessToVector<arr<arr<dbl, 3>, 3>> meta_coefficient(number_of_approx+1);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> cell_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> N_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_stress (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dbl, 3>, 3>, 3>> true_meta_coef (number_of_approx);
        printf("dfdfvdfv %d\n", slae.solution[0].size());
        for (auto &&a : meta_coefficient.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            e = 0.0;
        for (auto &&a : cell_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        printf("dfdfvdfv %d\n", cell_func.content[0][0][0][0].size());
        // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (st i = 0; i < slae.solution[0].size(); ++i)
        {
            if ((i % 3) == x) cell_func[arr<i32, 3>{0, 0, 0}][x][i] = 1.0;
            if ((i % 3) == y) cell_func[arr<i32, 3>{0, 0, 0}][y][i] = 1.0;
            if ((i % 3) == z) cell_func[arr<i32, 3>{0, 0, 0}][z][i] = 1.0;
        };
        // cell_func[arr<i32, 3>{0, 0, 0}][x][x] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][y][y] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][z][z] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (auto &&a : N_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());

        // auto mean_coefficient = 
        //     OnCell::calculate_mean_coefficients<3> (domain.dof_handler, element_matrix.C);
        // auto area_of_domain = 
        //     OnCell::calculate_area_of_domain<3> (domain.dof_handler);

        OnCell::MetaCoefficientElasticCalculator mc_calculator (
                domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());

        // for (auto &&approximation : approximations)
        // {
            // auto approximation = approximations[0];
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                slae.solution[0] = 0.0;
                                slae.rhsv[0] = 0.0;
                                // printf("scsdcdfvdf %d\n", approximation);

                                OnCell::SourceVectorApprox<3> element_rhsv (approximation, nu,
                                        element_matrix.C, 
                                        meta_coefficient,
                                        cell_func,
                                        // &psi_func,
                                        domain.dof_handler.get_fe());
                                // printf("%d\n", slae.rhsv[0].size());
                                OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);

                                printf("problem %ld %ld %ld %ld\n", i, j, k, nu);
                                printf("Integ %f\n", element_rhsv.tmp);
                                dealii::SolverControl solver_control (500000, 1e-12);
                                dealii::SolverCG<> solver (solver_control);
                                solver.solve (
                                        slae.matrix,
                                        slae.solution[0],
                                        slae.rhsv[0]
                                        ,dealii::PreconditionIdentity()
                                        );
                                FOR(i, 0, slae.solution[0].size())
                                    slae.solution[0][i] = slae.solution[0][bows.subst (i)];
                                FOR(i, 0, slae.rhsv[0].size())
                                    slae.rhsv[0][i] = slae.rhsv[0][bows.subst (i)];

                                cell_func[approximation][nu] = slae.solution[0];
                                // N_func[approximation][nu] = slae.rhsv[0];
                            };
                        };
                    };
                };
            };
            puts("!!!");
            for (st i = 0; i < approx_number+2; ++i)
            {
                for (st j = 0; j < approx_number+2; ++j)
                {
                    for (st k = 0; k < approx_number+2; ++k)
                    {
                        if ((i+j+k) == approx_number+1)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                auto res = mc_calculator .calculate (
                                        approximation, nu,
                                        domain.dof_handler, cell_func);
                                meta_coefficient[approximation][nu][x] = res[x]; //E_x_a[0]_nu_a[1]
                                meta_coefficient[approximation][nu][y] = res[y]; 
                                meta_coefficient[approximation][nu][z] = res[z]; 
                                printf("meta k=(%ld, %ld, %ld) nu=%ld %f %f %f\n", i, j, k, nu, 
                                meta_coefficient[approximation][nu][x],
                                meta_coefficient[approximation][nu][y],
                                meta_coefficient[approximation][nu][z]
                                        );
                            };
                        };
                    };
                };
            };
        };
        puts("!!!!!");

        {
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
            arr<i32, 3> approx = {1, 0, 0};
            printf("11111\n");
            stress_calculator .calculate (
                    approx, x, x, x,
                    domain.dof_handler, cell_func, stress);
            stress_calculator .calculate (
                    approx, x, x, y,
                    domain.dof_handler, cell_func, stress);
            // stress_calculator .calculate (
            //         approx, x, x, z,
            //         domain.dof_handler, cell_func, stress);
            EPTools ::print_move_slice (stress, domain.dof_handler, 
                    "stress_slice_approx_1_0_0_x_x.gpd", z, 0.5);
        };

        {
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
            arr<i32, 3> approx = {2, 0, 0};
            printf("11111\n");
            stress_calculator .calculate (
                    approx, x, x, x,
                    domain.dof_handler, cell_func, stress);
            EPTools ::print_move_slice (stress, domain.dof_handler, 
                    "stress_slice_approx_2x_x_x_x.gpd", z, 0.5);
            dbl Integrall = 0.0;
            for (st i = 0; i < stress.size(); ++i)
                Integrall += stress[i];
            printf("Integrall %f\n", Integrall);
            // {
            //     dealii::Vector<dbl> stress_diff_move(domain.dof_handler.n_dofs());
            //     for (st i = 0; i < stress.size(); ++i)
            //     {
            //         stress_diff_move[i] = (stress[i] + cell_func[arr<i32, 3>{1, 0, 0}][x][i]);
            //     };
            //     EPTools ::print_move_slice (stress_diff_move, domain.dof_handler, 
            //             "stress_slice_approx_2x_x_x_x_diff_move_xx.gpd", z, 0.5);
            // };
        };

                printf("\n");
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
            {
                for (st i = 0; i < approx_number+1; ++i)
                {
                    for (st j = 0; j < approx_number+1; ++j)
                    {
                        for (st k = 0; k < approx_number+1; ++k)
                        {
                            if ((i+j+k) == approx_number)
                            {
                                arr<i32, 3> approximation = {i, j, k};
                                for (st nu = 0; nu < 3; ++nu)
                                {
                                    for (st alpha = 0; alpha < 3; ++alpha)
                                    {
                                        dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
                                        for (st beta = 0; beta < 3; ++beta)
                                        {
                                            stress_calculator .calculate (
                                                    approximation, nu, alpha, beta,
                                                    domain.dof_handler, cell_func, stress);
                                            true_meta_coef[approximation][nu][alpha][beta] =
                                                OnCell::calculate_meta_coefficients_3d_elastic_from_stress (
                                                        domain.dof_handler, stress, beta);
                                printf("meta k=(%ld, %ld, %ld) nu=%ld alpha=%ld beta=%ld %f\n", 
                                        i, j, k, nu, alpha, beta, true_meta_coef[approximation][nu][alpha][beta]);
                                        };
                                        cell_stress[approximation][nu][alpha] = stress;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{1,0,0}][0][0], domain.dof_handler, 
                    "stress_slice_approx_x_x_x.gpd", y, 0.5);

        // printf("Meta_xxxx %f\n", meta_coefficient[arr<i32, 3>{2, 0, 0}][x][x]);
        // printf("Meta_yxxy %f\n", meta_coefficient[arr<i32, 3>{1, 1, 0}][x][y]);
        // printf("Meta_yyxx %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][x][x]);
        // printf("Meta_xzxz %f\n", meta_coefficient[arr<i32, 3>{1, 0, 1}][x][z]);
        // printf("Meta_zzzz %f\n", meta_coefficient[arr<i32, 3>{0, 0, 2}][z][z]);
        // printf("Meta_zzyy %f\n", meta_coefficient[arr<i32, 3>{0, 1, 1}][y][z]);
        // printf("Meta_zyzy %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][z][z]);
        ATools::FourthOrderTensor meta_coef;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                meta_coef[j][x][i][x] = meta_coefficient[arr<i32, 3>{2, 0, 0}][i][j];
                meta_coef[j][x][i][y] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][x][i][z] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][y][i][x] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][y][i][y] = meta_coefficient[arr<i32, 3>{0, 2, 0}][i][j];
                meta_coef[j][y][i][z] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][x] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][z][i][y] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][z] = meta_coefficient[arr<i32, 3>{0, 0, 2}][i][j];
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        auto newcoef = unphysical_to_physicaly (meta_coef);
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
        printf("\n");
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                for (st k = 0; k < 3; ++k)
                {
                    meta_coef[j][k][i][x] = true_meta_coef[arr<i32, 3>{1, 0, 0}][i][j][k];
                    meta_coef[j][k][i][y] = true_meta_coef[arr<i32, 3>{0, 1, 0}][i][j][k];
                    meta_coef[j][k][i][z] = true_meta_coef[arr<i32, 3>{0, 0, 1}][i][j][k];
                };
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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

        {
        std::ofstream out ("cell/meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ofstream out ("cell/stress_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/stress_"+name+".gpd", y, 0.5);
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                str name = ort[i]+str("_")+ort[j]+str("_")+ort[k]+str("_")+ort[nu];
                                {
                                    std::ofstream out ("cell/solution_"+name+".bin", std::ios::out | std::ios::binary);
                                    for (st i = 0; i < slae.solution[0].size(); ++i)
                                    {
                                        out.write ((char *) &(cell_func[approximation][nu][i]), sizeof(dbl));
                                    };
                                    out.close ();
                                };
                            };
                        };
                    };
                };
            };
        };

        {
            std::ofstream out ("cell/solution_on_cell_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution[0].size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };


        OnCell::ArrayWithAccessToVector<arr<str, 3>> file_name (number_of_approx);
        file_name[arr<i32, 3>{1, 0, 0}][x] = "move_slice_approx_xx.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][y] = "move_slice_approx_xy.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][z] = "move_slice_approx_xz.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][x] = "move_slice_approx_yx.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][y] = "move_slice_approx_yy.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][z] = "move_slice_approx_yz.gpd";
        EPTools ::print_coor_bin<3> (domain.dof_handler, "cell/coor_cell.bin");
    // {
    //     // puts("4444444444");
    //     printf("dofs %d\n", domain.dof_handler.n_dofs());
    //     vec<dealii::Point<dim>> coor(domain.dof_handler.n_dofs());
    //     puts("555555555");
    //     {
    //         cu8 dofs_per_cell =  domain.dof_handler.get_fe().dofs_per_cell;
    //         printf("dofs %d\n", dofs_per_cell);
    //
    //         std::vector<u32> local_dof_indices (dofs_per_cell);
    //
    //         auto cell = dof_handler.begin_active();
    //         auto endc = dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             cell ->get_dof_indices (local_dof_indices);
    //
    //             // FOR (i, 0, dofs_per_cell)
    //             //     indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
    //             FOR (i, 0, dofs_per_cell)
    //             {
    //                 coor[local_dof_indices[i]] = cell ->vertex (i);
    //             };
    //         };
    //     };
    //
    //     {
    //         std::ofstream out ("cell/coor_hole.bin", std::ios::out | std::ios::binary);
    //         for (st i = 0; i < coor.size(); ++i)
    //         {
    //             for (st j = 0; j < dim; ++j)
    //             {
    //                 out.write ((char *) &(coor[i](j)), sizeof(dbl));
    //             };
    //         };
    //         out.close ();
    //     };
    // };
        puts("222222222222222222222222222222");
        // for (auto &&approximation : approximations)
        // {
        //     for (st nu = 0; nu < 3; ++nu)
        //     {
        //         EPTools ::print_move_slice (cell_func[approximation][nu], domain.dof_handler, 
        //                 file_name[approximation][nu], z, 0.5);
        //     };
        // };
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][z], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][z], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{2, 0, 0}][x], domain.dof_handler, 
                "move_slice_approx_2x_x.gpd", z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 2, 0}][x], domain.dof_handler, 
                "move_slice_approx_2y_x.gpd", z, 0.5);
        puts("222222222222222222222222222222");

        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{1, 0, 0}][x][x], domain.dof_handler, 
                "stress_slice_xx.gpd", z, 0.5);
        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{0, 1, 0}][y][x], domain.dof_handler, 
                "stress_slice_yy.gpd", z, 0.5);

            // EPTools ::print_move<2> (slae.solution[0], domain.dof_handler, "move_approx");
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);

//         arr<u8, 4> theta  = {x, y, z, x};
//         arr<u8, 4> lambda = {x, y, z, y};
//
// #pragma omp parallel for
//         for (st n = 0; n < 4; ++n)
//         {
//             vec<arr<arr<dbl, 2>, 2>> coef_for_rhs(2);
//
//             for (auto i : {x, y})
//                 for (auto j : {x, y})
//                     for(st k = 0; k < element_matrix.C.size(); ++k)
//                     {
//                         coef_for_rhs[k][i][j] = 
//                             element_matrix.C[k][i][j][theta[n]][lambda[n]];
//                     };
//
//             slae.solution[n] = 0;
//             slae.rhsv[n] = 0;
//
//             OnCell::SourceVector<2> element_rhsv (
//                     coef_for_rhs, domain.dof_handler.get_fe());
//             OnCell::Assembler::assemble_rhsv<2> (
//                     slae.rhsv[n], element_rhsv, domain.dof_handler, bows);
//
//             dealii::SolverControl solver_control (10000, 1e-12);
//             dealii::SolverCG<> solver (solver_control);
//             solver.solve (
//                     slae.matrix,
//                     slae.solution[n],
//                     slae.rhsv[n]
//                     ,dealii::PreconditionIdentity()
//                     );
//             FOR(i, 0, slae.solution[n].size())
//                 slae.solution[n][i] = slae.solution[n][bows.subst (i)];
//         };
//
//         OnCell::SystemsLinearAlgebraicEquations<2> problem_of_torsion_rod_slae;
//         vec<ATools::SecondOrderTensor> coef_for_potr(2);
//         for (st i = 0; i < 2; ++i)
//         {
//             coef_for_potr[i][x][x] = element_matrix.C[i][x][z][x][z];
//             coef_for_potr[i][y][y] = element_matrix.C[i][y][z][y][z];
//             coef_for_potr[i][x][y] = element_matrix.C[i][x][z][y][z];
//             coef_for_potr[i][y][x] = element_matrix.C[i][x][z][y][z];
//         };
//         solve_heat_problem_on_cell_aka_torsion_rod<2> (
//                 domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);
//
//         arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
//         for (st i = 0; i < 4; ++i)
//         {
//             EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
//         };
//
//         auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
//                 domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);
//
//         for (size_t i = 0; i < 9; ++i)
//         {
//             uint8_t im = i / (dim + 1);
//             uint8_t in = i % (dim + 1);
//
//             for (size_t j = 0; j < 9; ++j)
//             {
//                 uint8_t jm = j / (dim + 1);
//                 uint8_t jn = j % (dim + 1);
//
//                 if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
//                     printf("\x1B[31m%f\x1B[0m   ", 
//                             meta_coef[im][in][jm][jn]);
//                 else
//                     printf("%f   ", 
//                             meta_coef[im][in][jm][jn]);
//             };
//             for (size_t i = 0; i < 2; ++i)
//                 printf("\n");
//         };
//         // print_tensor<6*6>(meta_coef);
//         // {
//         //     auto newcoef = unphysical_to_physicaly (meta_coef);
//         //     // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
//         //     printf("%f %f %f %f %f %f %f %f %f %f %f\n",
//         //             newcoef[0][0][0][0],
//         //             newcoef[0][0][1][1],
//         //             newcoef[0][0][2][2],
//         //             newcoef[1][1][0][0],
//         //             newcoef[1][1][1][1],
//         //             newcoef[1][1][2][2],
//         //             newcoef[2][2][0][0],
//         //             newcoef[2][2][1][1],
//         //             newcoef[2][2][2][2],
//         //             meta_coef[0][1][0][1],
//         //             meta_coef[0][2][0][2]
//         //           );
//         // };
    };
    };
void solve_two_stress (cst flag)
{
    if (flag)
    {  
        enum {x, y, z};

        cst ort_slice = y;
        cdbl coor_slice = 0.5;

        ATools::FourthOrderTensor C;
        std::ifstream in ("cell/meta_coef.bin", std::ios::in | std::ios::binary);
        in.read ((char *) &C, sizeof C);

        st size_sol_hole = 0;
        {
            std::ifstream in ("hole/solution_hole_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_hole, sizeof size_sol_hole);
            in.close ();
        };
        st size_sol_cell = 0;
        {
            std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_cell, sizeof size_sol_cell);
            in.close ();
        };
        arr<vec<dbl>, 2> deform_1;
        arr<arr<vec<dbl>, 2>, 2> deform_2;
        deform_1[x] .resize (size_sol_hole);
        deform_1[y] .resize (size_sol_hole);
        deform_2[x][x] .resize (size_sol_hole);
        deform_2[x][y] .resize (size_sol_hole);
        deform_2[y][x] .resize (size_sol_hole);
        deform_2[y][y] .resize (size_sol_hole);
        {
            std::ifstream in ("hole/deform_hole_x.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_y.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/stress_hole_y.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_xx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_xy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_yx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_yy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][y][i], sizeof(dbl));
            };
            in.close ();
        };

        cst number_of_approx = 3;

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_stress (number_of_approx);
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        // std::cout << "cell/stress_"+name+".bin" << std::endl;
                                        std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::cout << in.is_open() << std::endl;
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_stress[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        // {
        //     // std::cout << "cell/stress_"+name+".bin" << std::endl;
        //     std::ifstream in ("cell/stress_1_0_0_x_x.bin", std::ios::in | std::ios::binary);
        //     // std::cout << in.is_open() << std::endl;
        //     arr<dbl, 3> tmp;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
        //         dbl tmp = 0.0;
        //         in.read ((char *) &(tmp), sizeof(dbl));
        //         cell_stress[arr<i32, 3>{1, 0, 0}][x][x][i] = tmp;
        //     };
        //     in.close ();
        // };

        vec<dealii::Point<2>> coor_hole (size_sol_hole);
        {
            std::ifstream in ("hole/coor_hole.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &coor_hole[i](x), sizeof(dbl));
                in.read ((char *) &coor_hole[i](y), sizeof(dbl));
            };
            in.close ();
        };

        vec<dealii::Point<3>> coor_cell (size_sol_cell);
        {
            std::ifstream in ("cell/coor_hole.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_cell; ++i)
            {
                in.read ((char *) &coor_cell[i](x), sizeof(dbl));
                in.read ((char *) &coor_cell[i](y), sizeof(dbl));
                in.read ((char *) &coor_cell[i](z), sizeof(dbl));
            };
            in.close ();
        };
        {
            FILE *F;
            F = fopen("stress_cell_yyy.gpd", "w");
        for (st i = 0; i < size_sol_cell; ++i)
        {
            if (i % 3 == y)
            fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
        };
            fclose(F);
        };
        // for (st i = 0; i < 10; ++i)
        // {
        //     printf("%f %f %f\n", coor_cell[i](x), coor_cell[i](y), coor_cell[i](z));
        // };
        
        st size_line_hole = 0;
        {
            FILE *F;
            F = fopen("deform_1_yy_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == y)
                    {
                fprintf(F,"%f %f\n", coor_hole[i](x), deform_1[y][i]);
                // fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[x][y][i]);
                ++size_line_hole;
                // printf("%f\n", deform_1[y][i]);
                    };
                };
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_1_yyx_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == x)
                    {
                        fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[y][y][i]);
                    };
                };
            };
            fclose(F);
        };
        st size_line_cell = 0;
        {
            FILE *F;
            F = fopen("stress_cell_line.gpd", "w");
            for (st i = 0; i < size_sol_cell; ++i)
            {
                if (
                        (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10) and
                        (std::abs(coor_cell[i](z) - 0.5) < 1.0e-10)
                   )

                {
                    if ((i % 3) == y)
                    {
                fprintf(F,"%f %f\n", coor_cell[i](x), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
                ++size_line_cell;
                    };
                };
            };
            fclose(F);
        };

        arr<arr<vec<dbl>, 2>, 2> deform_line_1;
        arr<arr<arr<vec<dbl>, 2>, 2>, 2> deform_line_2;
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               deform_line_1[i][j] .resize (size_line_hole);
               
           }; 
        };
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               for (st k = 0; k < 2; ++k)
               {
               deform_line_2[i][j][k] .resize (size_line_hole);
               };
           }; 
        };
        vec<dealii::Point<2>> coor_line_hole (size_line_hole);
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            if ((i % 2) == k)
                            {
                                deform_line_1[k][j][n] = deform_1[j][i];
                                // printf("%d\n", n);
                            };
                        };
                    };
                    if ((i % 2))
                    {
                coor_line_hole[n] = coor_hole[i];
                ++n;
                    };
                };
            };
        };
        {
            FILE *F;
            F = fopen("deform_1_yy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_1[y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_1_xx_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_1[x][x][i]);
            };
            fclose(F);
        };
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            for (st l = 0; l < 2; ++l)
                            {
                                if ((i % 2) == k)
                                {
                                    deform_line_2[k][j][l][n] = deform_2[j][l][i];
                                };
                            };
                        };
                    };
                    if ((i % 2))
                    {
                ++n;
                    };
                };
            };
        };
        {
            FILE *F;
            F = fopen("deform_2_yyx_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](x), deform_line_2[y][y][x][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_2_yyy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_2[y][y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_2_xxy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_2[x][x][y][i]);
            };
            fclose(F);
        };
        //     // for (st i = 0; i < size_line_hole; ++i)
        //     // {
        //     //     printf("%f %f\n", coor_line_hole[i](x), coor_line_hole[i](y));
        //     // };
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_line_stress (number_of_approx);
        for (auto &&a : cell_line_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_line_cell);
        vec<dealii::Point<3>> coor_line_cell (size_line_cell);
        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10) and
                        (std::abs(coor_cell[m](z) - 0.5) < 1.0e-10)
                   )

                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                        cell_line_stress[approximation][nu][alpha][beta][n] = 
                                                             cell_stress[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    if ((m % 3) == z)
                    {
                        coor_line_cell[n] = coor_cell[m];
                        ++n;
                    };
                    if (n == size_line_cell)
                        break;
                };
            };
        };
        // {
        //     st n = 0;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (
        //                 (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10) and
        //                 (std::abs(coor_cell[i](z) - 0.5) < 1.0e-10)
        //            )
        //
        //         {
        //             st beta = y;
        //             {
        //                 if ((i % 3) == beta)
        //                 {
        //                     cell_line_stress[arr<i32, 3>{0,1,0}][y][y][y][n] = 
        //                          cell_stress[arr<i32, 3>{0,1,0}][y][y][i];
        //                 ++n;
        //                 };
        //             };
        //         };
        //     };
        // };
        // {
        //     FILE *F;
        //     F = fopen("stress_cell_flat.gpd", "w");
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (
        //                 (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10)
        //            )
        //
        //         {
        //             if ((i % 3) == 0)
        //         fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{1,0,0}][x][x][i]);
        //         };
        //     };
        //     fclose(F);
        // };
        {
            FILE *F;
            F = fopen("stress_cell_line_2.gpd", "w");
            for (st i = 0; i < size_line_cell; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][i]);
                // printf("%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("stress_cell_line_2.gpd", "w");
            for (st i = 0; i < size_line_cell; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][i]);
                // printf("%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][i]);
            };
            fclose(F);
        };

        printf("size_line %d %d\n", size_line_hole, size_line_cell);
        printf("size %d %d\n", size_sol_hole, size_sol_cell);

        cst num_cells = 100;
        cdbl cell_size = 1.0 / num_cells;
        for (st i = 0; i < size_line_cell; ++i)
        {
            coor_line_cell[i](x) /= num_cells;
        };


        cst fgh = 10000;
        arr<arr<vec<dbl>, 3>, 3> macro_stress;
        arr<arr<vec<dbl>, 3>, 3> final_stress;
        arr<arr<vec<dbl>, 3>, 3> final_stress_2;
        for (st i = 0; i < 3; ++i)
        {
           for (st j = 0; j < 3; ++j)
           {
               // macro_stress[i][j] .resize (size_line_hole);
               // final_stress[i][j] .resize (size_line_hole);
               // final_stress_2[i][j] .resize (size_line_hole);
               macro_stress[i][j] .resize (fgh);
               final_stress[i][j] .resize (fgh);
               final_stress_2[i][j] .resize (fgh);
           }; 
        };
        {
            FILE *F;
            F = fopen("hole_plas_cell.gpd", "w");
            // for (st i = 0; i < size_line_hole; ++i)
            for (st i = 0; i < fgh+1; ++i)
            {
                // dbl coor_in_cell = coor_line_hole[i](x);
                dbl coor_in_hole = 1.0 / fgh * i;
                dbl coor_in_cell = coor_in_hole;
                for (st j = 0; j < num_cells; ++j)
                {
                    // printf("%f\n", coor_in_cell);
                    coor_in_cell -= cell_size;
                    if (coor_in_cell < 0.0)
                    {
                        coor_in_cell += cell_size;
                        break;
                    };
                };
                // printf("%f %f %f\n", 
                //         1.0 / fgh * i,
                //         coor_in_cell, cell_size);
                // printf("\n");
                // dbl sol_in_cell = 0.0;
                st num_of_point_in_cell = 0;
                for (st j = 0; j < size_line_cell; ++j)
                {
                    if (coor_in_cell < coor_line_cell[j](x))
                    {
                        // dbl X = coor_in_cell;
                        // dbl X1 = coor_line_cell[j-1](x);
                        // dbl X2 = coor_line_cell[j](x);
                        // dbl Y1 = cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][j-1];
                        // dbl Y2 = cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][j];
                        // sol_in_cell = (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        num_of_point_in_cell = j;
                        // printf("%f %f %f\n", coor_line_hole[i](x), coor_in_cell, coor_line_cell[j](x));
                        break;
                    };
                };
                st num_of_point_in_hole = 0;
                for (st j = 0; j < size_line_hole; ++j)
                {
                    if ((coor_in_hole) < coor_line_hole[j](x))
                    {
                        num_of_point_in_hole = j;
                        break;
                    };
                };
                auto sol_in_cell  = [&coor_in_cell, &coor_line_cell, &cell_line_stress, num_of_point_in_cell] 
                    (cst i, cst j, cst k, cst l, cst m, cst n)
                    {
                        cst nm = num_of_point_in_cell;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_cell.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_cell;
                        cdbl X1 = coor_line_cell[nm_1](0);
                        cdbl X2 = coor_line_cell[nm](0);
                        cdbl Y1 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm_1];
                        cdbl Y2 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm];
                        // printf("%f %f %f %f %f %f %ld %ld %f\n",
                        //         (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2), X, X1, X2, Y1, Y2, nm_1, nm,
                        //         cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][0]);
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                auto sol_in_hole  = [&coor_in_hole, &coor_line_hole, &deform_line_1, num_of_point_in_hole] 
                    (cst i, cst j)
                    {
                        cst nm = num_of_point_in_hole;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_hole.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_hole;
                        cdbl X1 = coor_line_hole[nm_1](0);
                        cdbl X2 = coor_line_hole[nm](0);
                        cdbl Y1 = deform_line_1[i][j][nm_1];
                        cdbl Y2 = deform_line_1[i][j][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                // cst num = num_of_point_in_cell;

                for (st alpha = 0; alpha < 3; ++alpha)
                {
                    for (st beta = 0; beta < 3; ++beta)
                    {
                       macro_stress[alpha][beta][i] = 
                           C[x][x][alpha][beta] * sol_in_hole(x,x) +
                           C[y][x][alpha][beta] * sol_in_hole(y,x) +
                           C[x][y][alpha][beta] * sol_in_hole(x,y) +
                           C[y][y][alpha][beta] * sol_in_hole(y,y);
                       final_stress[alpha][beta][i] = 
                           sol_in_cell(1,0,0,x,alpha,beta) * sol_in_hole(x,x) +
                           sol_in_cell(1,0,0,y,alpha,beta) * sol_in_hole(y,x) +
                           sol_in_cell(0,1,0,x,alpha,beta) * sol_in_hole(x,y) +
                           sol_in_cell(0,1,0,y,alpha,beta) * sol_in_hole(y,y);
                    };
                };
                // for (st alpha = 0; alpha < 3; ++alpha)
                // {
                //     for (st beta = 0; beta < 3; ++beta)
                //     {
                //        macro_stress[alpha][beta][i] = 
                //            C[x][x][alpha][beta] * deform_line_1[x][x][i] +
                //            C[y][x][alpha][beta] * deform_line_1[y][x][i] +
                //            C[x][y][alpha][beta] * deform_line_1[x][y][i] +
                //            C[y][y][alpha][beta] * deform_line_1[y][y][i];
                //        final_stress[alpha][beta][i] = 
                //            sol_in_cell(1,0,0,x,alpha,beta, num) * deform_line_1[x][x][i] +
                //            sol_in_cell(1,0,0,y,alpha,beta, num) * deform_line_1[y][x][i] +
                //            sol_in_cell(0,1,0,x,alpha,beta, num) * deform_line_1[x][y][i] +
                //            sol_in_cell(0,1,0,y,alpha,beta, num) * deform_line_1[y][y][i];
                //        final_stress_2[alpha][beta][i] = 
                //            sol_in_cell(1,0,0,x,alpha,beta, num) * deform_line_1[x][x][i] +
                //            sol_in_cell(1,0,0,y,alpha,beta, num) * deform_line_1[y][x][i] +
                //            sol_in_cell(0,1,0,x,alpha,beta, num) * deform_line_1[x][y][i] +
                //            sol_in_cell(0,1,0,y,alpha,beta, num) * deform_line_1[y][y][i] +
                //            (
                //            sol_in_cell(2,0,0,x,alpha,beta, num) * deform_line_2[x][x][x][i] +
                //            sol_in_cell(1,1,0,x,alpha,beta, num) * deform_line_2[x][y][x][i] +
                //            sol_in_cell(1,1,0,x,alpha,beta, num) * deform_line_2[x][x][y][i] +
                //            sol_in_cell(0,2,0,x,alpha,beta, num) * deform_line_2[x][y][y][i] +
                //            sol_in_cell(2,0,0,y,alpha,beta, num) * deform_line_2[y][x][x][i] +
                //            sol_in_cell(1,1,0,y,alpha,beta, num) * deform_line_2[y][y][x][i] +
                //            sol_in_cell(1,1,0,y,alpha,beta, num) * deform_line_2[y][x][y][i] +
                //            sol_in_cell(0,2,0,y,alpha,beta, num) * deform_line_2[y][y][y][i]
                //            ) * cell_size;
                //     };
                // };
                //
                fprintf(F, "%f %f %f %f %f\n", 
                        coor_in_hole, 
                        sol_in_cell(0,1,0,y,y,y), sol_in_hole(y,y), 
                        final_stress[y][y][i], macro_stress[y][y][i]);
                // fprintf(F, "%f %f %f %f %f %f %f %f\n", 
                //         coor_line_hole[i](x), 
                //         deform_line_1[y][y][i], deform_line_2[y][y][x][i],
                //         sol_in_cell(0,1,0,y,y,y,num), sol_in_cell(0,2,0,y,x,x,num), 
                //         final_stress[y][y][i], final_stress_2[y][y][i], macro_stress[y][y][i]);
                // fprintf(F, "%f %f %f\n", 
                //         coor_in_hole,
                //         coor_in_cell,
                //         sol_in_cell(0,1,0,y,y,y));
                // fprintf(F, "%f %f %f %f %f %f %f\n", 
                //         coor_line_hole[i](y), deform_line_1[x][x][i], deform_line_2[x][x][y][i],
                //         sol_in_cell(1,0,0,x,x,x,num), sol_in_cell(2,0,0,x,y,y,num), 
                //         final_stress[x][x][i], final_stress_2[x][x][i]);
        // for (st o = 0; o < size_line_cell; ++o)
        // {
        //     printf("%f %f %f\n", cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][o],
        //             sol_in_cell(0,1,0,y,y,y,o), coor_in_cell);
        // };
        // puts(" ");
            };
            fclose(F);
        };
        dbl max_macro_stress = 0.0;
        dbl max_final_stress = 0.0;
        for (st i = 0; i < fgh; ++i)
        {
            if (max_macro_stress < macro_stress[y][y][i])
                max_macro_stress = macro_stress[y][y][i];
            if (max_final_stress < final_stress[y][y][i])
                max_final_stress = final_stress[y][y][i];
        };
        {
            FILE *F;
            F = fopen("max_25.gpd", "a");
            fprintf(F, " %f %f\n", max_macro_stress, max_final_stress);
            fclose(F);
        };
    };
};

void solve_approx_cell_elastic_problem (cst flag, cdbl E, cdbl pua)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<3> domain;
        {
            // set_cylinder(domain.grid, 0.25, y, 2);
            set_cylinder_true(domain.grid, 0.49, z, 40, 5);
            // set_ball(domain.grid, 0.4, 3);
            // set_rect_3d(domain.grid,
            //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
            //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
        };
        dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        OnCell::SystemsLinearAlgebraicEquations<1> slae;
        OnCell::BlackOnWhiteSubstituter bows;
        OnCell::SystemsLinearAlgebraicEquations<1> slae_2;
        OnCell::BlackOnWhiteSubstituter bows_2;

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(element_matrix.C[1]);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : pua}(element_matrix.C[0]);
        // // {
        // // auto C2d = t4_to_t2 (element_matrix.C[0]);
        // //     printf("C2d\n");
        // //     for (size_t i = 0; i < 6; ++i)
        // //     {
        // //         for (size_t j = 0; j < 6; ++j)
        // //         {
        // //             if (std::abs(C2d[i][j]) > 0.0000001)
        // //                 printf("\x1B[31m%f\x1B[0m   ", 
        // //                         C2d[i][j]);
        // //             else
        // //                 printf("%f   ", 
        // //                         C2d[i][j]);
        // //         };
        // //         for (size_t i = 0; i < 2; ++i)
        // //             printf("\n");
        // //     };
        // //     printf("\n");
        // // };
        EPTools ::set_isotropic_elascity{yung : E, puasson : 0.25}(element_matrix.C[1]);
        
        // {
        //     arr<arr<dbl, 6>, 6> E2d_original;
        //     // arr<arr<dbl, 6>, 6> E2d_final;
        //     for (st i = 0; i < 6; ++i)
        //     {
        //         for (st j = 0; j < 6; ++j)
        //         {
        //             E2d_original[i][j] = 0.0;
        //         };
        //     };
        //
        //     arr<dbl, 3> E = {10.0, 10.0, 10.0};
        //
        //     E2d_original[0][0] = 1.0 / E[0];
        //     E2d_original[1][1] = 1.0 / E[1];
        //     E2d_original[2][2] = 1.0 / E[2];
        //     E2d_original[0][1] = -0.25 / E[0];
        //     E2d_original[0][2] = -0.25 / E[0];
        //     E2d_original[1][0] = -0.25 / E[1];
        //     E2d_original[1][2] = -0.25 / E[1];
        //     E2d_original[2][0] = -0.25 / E[2];
        //     E2d_original[2][1] = -0.25 / E[2];
        //     E2d_original[3][3] = 1.0 / 0.4;
        //     E2d_original[4][4] = 1.0 / 0.4;
        //     E2d_original[5][5] = 1.0 / 0.4;
        //
        //     auto C2d_original = inverse (E2d_original);
        //     auto C = t2_to_t4 (C2d_original);
        //     for (st i = 0; i < 3; ++i)
        //     {
        //         for (st j = 0; j < 3; ++j)
        //         {
        //             for (st k = 0; k < 3; ++k)
        //             {
        //                 for (st l = 0; l < 3; ++l)
        //                 {
        //                     element_matrix.C[1][i][j][k][l] = C[i][j][k][l];
        //                 };
        //             };
        //         };
        //     };
        // };

        // OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);
        OnCell::prepare_system_equations_alternate<3, 3, 1> (slae, bows, domain);
        // OnCell::prepare_system_equations_alternate<3, 3, 1> (slae_2, bows_2, domain);
        // std::cout << bows.size << " " << bows_2.size << std::endl;
        // {
        //     std::ofstream f("/home/primat/tmp/bows1.txt", std::ios::out);
        //     for (st i = 0; i < bows.size; ++i)
        //     {
        //         f << bows.white[i] << " " << bows.black[i] << std::endl;
        //     };
        //     f.close ();
        // };
        // {
        //     std::ofstream f("/home/primat/tmp/bows2.txt", std::ios::out);
        //     for (st i = 0; i < bows_2.size; ++i)
        //     {
        //         f << bows_2.white[i] << " " << bows_2.black[i] << std::endl;
        //     };
        //     f.close ();
        // };
        // puts("???????????????????");
        // for (st i = 0; i < bows_2.size; ++i)
        // {
        //     std::cout << bows_2.white[i] << " " << bows_2.black[i] << std::endl;
        // };
        // for (st i = 0; i < 81; ++i)
        // {
        //     for (st j = 0; j < 81; ++j)
        //     {
        //         std::cout << slae.sparsity_pattern.exists(i,j) << "";
        //     };
        //     std::cout << std::endl;
        // };
        // std::cout << std::endl;
        // for (st i = 0; i < 81; ++i)
        // {
        //     for (st j = 0; j < 81; ++j)
        //     {
        //         std::cout << slae_2.sparsity_pattern.exists(i,j) << "";
        //     };
        //     std::cout << std::endl;
        // };
        // std::cout << std::endl;
        // {
        //     std::ofstream f("/home/primat/tmp/sp1.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // {
        //     std::ofstream f("/home/primat/tmp/sp2.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae_2.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // exit(1);

        OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

        cst number_of_approx = 2; // Начиная с нулевой
        // arr<arr<i32, 3>, number_of_approx> approximations = {
        //     arr<i32, 3>({1, 0, 0}),
        //     arr<i32, 3>({0, 1, 0})};
            // arr<i32, 3>{2, 0, 0}};
        OnCell::ArrayWithAccessToVector<arr<arr<dbl, 3>, 3>> meta_coefficient(number_of_approx+1);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> cell_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> N_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_stress (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dbl, 3>, 3>, 3>> true_meta_coef (number_of_approx);
        printf("dfdfvdfv %d\n", slae.solution[0].size());
        for (auto &&a : meta_coefficient.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            e = 0.0;
        for (auto &&a : cell_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        printf("dfdfvdfv %d\n", cell_func.content[0][0][0][0].size());
        // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (st i = 0; i < slae.solution[0].size(); ++i)
        {
            if ((i % 3) == x) cell_func[arr<i32, 3>{0, 0, 0}][x][i] = 1.0;
            if ((i % 3) == y) cell_func[arr<i32, 3>{0, 0, 0}][y][i] = 1.0;
            if ((i % 3) == z) cell_func[arr<i32, 3>{0, 0, 0}][z][i] = 1.0;
        };
        // cell_func[arr<i32, 3>{0, 0, 0}][x][x] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][y][y] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][z][z] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (auto &&a : N_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());

        // auto mean_coefficient = 
        //     OnCell::calculate_mean_coefficients<3> (domain.dof_handler, element_matrix.C);
        // auto area_of_domain = 
        //     OnCell::calculate_area_of_domain<3> (domain.dof_handler);

        OnCell::MetaCoefficientElasticCalculator mc_calculator (
                domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());

        // for (auto &&approximation : approximations)
        // {
            // auto approximation = approximations[0];
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                slae.solution[0] = 0.0;
                                slae.rhsv[0] = 0.0;
                                // printf("scsdcdfvdf %d\n", approximation);

                                OnCell::SourceVectorApprox<3> element_rhsv (approximation, nu,
                                        element_matrix.C, 
                                        meta_coefficient,
                                        cell_func,
                                        // &psi_func,
                                        domain.dof_handler.get_fe());
                                // printf("%d\n", slae.rhsv[0].size());
                                OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);

                                printf("problem %ld %ld %ld %ld\n", i, j, k, nu);
                                printf("Integ %f\n", element_rhsv.tmp);
                                dealii::SolverControl solver_control (500000, 1e-12);
                                dealii::SolverCG<> solver (solver_control);
                                solver.solve (
                                        slae.matrix,
                                        slae.solution[0],
                                        slae.rhsv[0]
                                        ,dealii::PreconditionIdentity()
                                        );
                                FOR(i, 0, slae.solution[0].size())
                                    slae.solution[0][i] = slae.solution[0][bows.subst (i)];
                                FOR(i, 0, slae.rhsv[0].size())
                                    slae.rhsv[0][i] = slae.rhsv[0][bows.subst (i)];

                                cell_func[approximation][nu] = slae.solution[0];
                                // N_func[approximation][nu] = slae.rhsv[0];
                            };
                        };
                    };
                };
            };
            puts("!!!");
            for (st i = 0; i < approx_number+2; ++i)
            {
                for (st j = 0; j < approx_number+2; ++j)
                {
                    for (st k = 0; k < approx_number+2; ++k)
                    {
                        if ((i+j+k) == approx_number+1)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                auto res = mc_calculator .calculate (
                                        approximation, nu,
                                        domain.dof_handler, cell_func);
                                meta_coefficient[approximation][nu][x] = res[x]; //E_x_a[0]_nu_a[1]
                                meta_coefficient[approximation][nu][y] = res[y]; 
                                meta_coefficient[approximation][nu][z] = res[z]; 
                                printf("meta k=(%ld, %ld, %ld) nu=%ld %f %f %f\n", i, j, k, nu, 
                                meta_coefficient[approximation][nu][x],
                                meta_coefficient[approximation][nu][y],
                                meta_coefficient[approximation][nu][z]
                                        );
                            };
                        };
                    };
                };
            };
        };
        puts("!!!!!");

        {
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
            arr<i32, 3> approx = {1, 0, 0};
            printf("11111\n");
            stress_calculator .calculate (
                    approx, x, x, x,
                    domain.dof_handler, cell_func, stress);
            stress_calculator .calculate (
                    approx, x, x, y,
                    domain.dof_handler, cell_func, stress);
            // stress_calculator .calculate (
            //         approx, x, x, z,
            //         domain.dof_handler, cell_func, stress);
            EPTools ::print_move_slice (stress, domain.dof_handler, 
                    "stress_slice_approx_1_0_0_x_x.gpd", z, 0.5);
        };

        // {
        //     OnCell::StressCalculator stress_calculator (
        //             domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
        //     dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
        //     arr<i32, 3> approx = {2, 0, 0};
        //     printf("11111\n");
        //     stress_calculator .calculate (
        //             approx, x, x, x,
        //             domain.dof_handler, cell_func, stress);
        //     EPTools ::print_move_slice (stress, domain.dof_handler, 
        //             "stress_slice_approx_2x_x_x_x.gpd", z, 0.5);
        //     dbl Integrall = 0.0;
        //     for (st i = 0; i < stress.size(); ++i)
        //         Integrall += stress[i];
        //     printf("Integrall %f\n", Integrall);
        //     // {
        //     //     dealii::Vector<dbl> stress_diff_move(domain.dof_handler.n_dofs());
        //     //     for (st i = 0; i < stress.size(); ++i)
        //     //     {
        //     //         stress_diff_move[i] = (stress[i] + cell_func[arr<i32, 3>{1, 0, 0}][x][i]);
        //     //     };
        //     //     EPTools ::print_move_slice (stress_diff_move, domain.dof_handler, 
        //     //             "stress_slice_approx_2x_x_x_x_diff_move_xx.gpd", z, 0.5);
        //     // };
        // };

                printf("\n");
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
            {
                for (st i = 0; i < approx_number+1; ++i)
                {
                    for (st j = 0; j < approx_number+1; ++j)
                    {
                        for (st k = 0; k < approx_number+1; ++k)
                        {
                            if ((i+j+k) == approx_number)
                            {
                                arr<i32, 3> approximation = {i, j, k};
                                for (st nu = 0; nu < 3; ++nu)
                                {
                                    for (st alpha = 0; alpha < 3; ++alpha)
                                    {
                                        dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
                                        for (st beta = 0; beta < 3; ++beta)
                                        {
                                            stress_calculator .calculate (
                                                    approximation, nu, alpha, beta,
                                                    domain.dof_handler, cell_func, stress);
                                            true_meta_coef[approximation][nu][alpha][beta] =
                                                OnCell::calculate_meta_coefficients_3d_elastic_from_stress (
                                                        domain.dof_handler, stress, beta);
                                printf("meta k=(%ld, %ld, %ld) nu=%ld alpha=%ld beta=%ld %f\n", 
                                        i, j, k, nu, alpha, beta, true_meta_coef[approximation][nu][alpha][beta]);
                                        };
                                        cell_stress[approximation][nu][alpha] = stress;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{1,0,0}][0][0], domain.dof_handler, 
                    "stress_slice_approx_x_x_x.gpd", z, 0.5);

        // printf("Meta_xxxx %f\n", meta_coefficient[arr<i32, 3>{2, 0, 0}][x][x]);
        // printf("Meta_yxxy %f\n", meta_coefficient[arr<i32, 3>{1, 1, 0}][x][y]);
        // printf("Meta_yyxx %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][x][x]);
        // printf("Meta_xzxz %f\n", meta_coefficient[arr<i32, 3>{1, 0, 1}][x][z]);
        // printf("Meta_zzzz %f\n", meta_coefficient[arr<i32, 3>{0, 0, 2}][z][z]);
        // printf("Meta_zzyy %f\n", meta_coefficient[arr<i32, 3>{0, 1, 1}][y][z]);
        // printf("Meta_zyzy %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][z][z]);
        ATools::FourthOrderTensor meta_coef;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                meta_coef[j][x][i][x] = meta_coefficient[arr<i32, 3>{2, 0, 0}][i][j];
                meta_coef[j][x][i][y] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][x][i][z] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][y][i][x] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][y][i][y] = meta_coefficient[arr<i32, 3>{0, 2, 0}][i][j];
                meta_coef[j][y][i][z] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][x] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][z][i][y] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][z] = meta_coefficient[arr<i32, 3>{0, 0, 2}][i][j];
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        auto newcoef = unphysical_to_physicaly (meta_coef);
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
        // printf("\n");
        // auto C2d = t4_to_t2 (meta_coef);
        // auto E2d = inverse (C2d);
        // auto E = t2_to_t4 (E2d);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         1.0/E[0][0][0][0],
        //         -E[0][0][1][1]/E[0][0][0][0],
        //         -E[0][0][2][2]/E[0][0][0][0],
        //         -E[1][1][0][0]/E[1][1][1][1],
        //         1.0/E[1][1][1][1],
        //         -E[1][1][2][2]/E[1][1][1][1],
        //         -E[2][2][0][0]/E[2][2][2][2],
        //         -E[2][2][1][1]/E[2][2][2][2],
        //         1.0/E[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                for (st k = 0; k < 3; ++k)
                {
                    meta_coef[j][k][i][x] = true_meta_coef[arr<i32, 3>{1, 0, 0}][i][j][k];
                    meta_coef[j][k][i][y] = true_meta_coef[arr<i32, 3>{0, 1, 0}][i][j][k];
                    meta_coef[j][k][i][z] = true_meta_coef[arr<i32, 3>{0, 0, 1}][i][j][k];
                };
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        // auto newcoef = unphysical_to_physicaly (meta_coef);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         newcoef[0][0][0][0],
        //         newcoef[0][0][1][1],
        //         newcoef[0][0][2][2],
        //         newcoef[1][1][0][0],
        //         newcoef[1][1][1][1],
        //         newcoef[1][1][2][2],
        //         newcoef[2][2][0][0],
        //         newcoef[2][2][1][1],
        //         newcoef[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        // auto C2d = t4_to_t2 (meta_coef);
        // auto E2d = inverse (C2d);
        // auto Ef = t2_to_t4 (E2d);
        // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
        //         1.0/Ef[0][0][0][0],
        //         -Ef[0][0][1][1]/Ef[0][0][0][0],
        //         -Ef[0][0][2][2]/Ef[0][0][0][0],
        //         -Ef[1][1][0][0]/Ef[1][1][1][1],
        //         1.0/Ef[1][1][1][1],
        //         -Ef[1][1][2][2]/Ef[1][1][1][1],
        //         -Ef[2][2][0][0]/Ef[2][2][2][2],
        //         -Ef[2][2][1][1]/Ef[2][2][2][2],
        //         1.0/Ef[2][2][2][2],
        //         1.0/E2d[3][3],
        //         1.0/E2d[4][4],
        //         1.0/E2d[5][5],
        //         (C2d[0][0]-C2d[0][1])/2.0
        //         );
        // printf("\n");
        // {
        //     std::ofstream f("not_isotrop_125.gpd", std::ios::out | std::ios::app);
        //     f << E << " " << (C2d[0][0]-C2d[0][1])/2.0 << " " << 1.0/E2d[5][5] << std::endl;
        //     f.close();
        // };
        //
        //     printf("E2d_final\n");
        //     for (size_t i = 0; i < 6; ++i)
        //     {
        //         for (size_t j = 0; j < 6; ++j)
        //         {
        //             if (std::abs(E2d[i][j]) > 0.0000001)
        //                 printf("\x1B[31m%f\x1B[0m   ", 
        //                         E2d[i][j]);
        //             else
        //                 printf("%f   ", 
        //                         E2d[i][j]);
        //         };
        //         for (size_t i = 0; i < 2; ++i)
        //             printf("\n");
        //     };
        //     printf("\n");
        //
        //     printf("C2d_final\n");
        //     for (size_t i = 0; i < 6; ++i)
        //     {
        //         for (size_t j = 0; j < 6; ++j)
        //         {
        //             if (std::abs(C2d[i][j]) > 0.0000001)
        //                 printf("\x1B[31m%f\x1B[0m   ", 
        //                         C2d[i][j]);
        //             else
        //                 printf("%f   ", 
        //                         C2d[i][j]);
        //         };
        //         for (size_t i = 0; i < 2; ++i)
        //             printf("\n");
        //     };
        //     printf("\n");

        {
        std::ofstream out ("cell/meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ofstream out ("cell/stress_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/stress_"+name+".gpd", z, 0.5);
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                str name = ort[i]+str("_")+ort[j]+str("_")+ort[k]+str("_")+ort[nu];
                                {
                                    std::ofstream out ("cell/solution_"+name+".bin", std::ios::out | std::ios::binary);
                                    for (st i = 0; i < slae.solution[0].size(); ++i)
                                    {
                                        out.write ((char *) &(cell_func[approximation][nu][i]), sizeof(dbl));
                                    };
                                    out.close ();
                                };
                            };
                        };
                    };
                };
            };
        };

        {
            std::ofstream out ("cell/solution_on_cell_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution[0].size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };
            


        OnCell::ArrayWithAccessToVector<arr<str, 3>> file_name (number_of_approx);
        file_name[arr<i32, 3>{1, 0, 0}][x] = "move_slice_approx_xx.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][y] = "move_slice_approx_xy.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][z] = "move_slice_approx_xz.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][x] = "move_slice_approx_yx.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][y] = "move_slice_approx_yy.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][z] = "move_slice_approx_yz.gpd";
        EPTools ::print_coor_bin<3> (domain.dof_handler, "cell/coor_cell.bin");
    // {
    //     // puts("4444444444");
    //     printf("dofs %d\n", domain.dof_handler.n_dofs());
    //     vec<dealii::Point<dim>> coor(domain.dof_handler.n_dofs());
    //     puts("555555555");
    //     {
    //         cu8 dofs_per_cell =  domain.dof_handler.get_fe().dofs_per_cell;
    //         printf("dofs %d\n", dofs_per_cell);
    //
    //         std::vector<u32> local_dof_indices (dofs_per_cell);
    //
    //         auto cell = dof_handler.begin_active();
    //         auto endc = dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             cell ->get_dof_indices (local_dof_indices);
    //
    //             // FOR (i, 0, dofs_per_cell)
    //             //     indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
    //             FOR (i, 0, dofs_per_cell)
    //             {
    //                 coor[local_dof_indices[i]] = cell ->vertex (i);
    //             };
    //         };
    //     };
    //
    //     {
    //         std::ofstream out ("cell/coor_hole.bin", std::ios::out | std::ios::binary);
    //         for (st i = 0; i < coor.size(); ++i)
    //         {
    //             for (st j = 0; j < dim; ++j)
    //             {
    //                 out.write ((char *) &(coor[i](j)), sizeof(dbl));
    //             };
    //         };
    //         out.close ();
    //     };
    // };
        puts("222222222222222222222222222222");
        // for (auto &&approximation : approximations)
        // {
        //     for (st nu = 0; nu < 3; ++nu)
        //     {
        //         EPTools ::print_move_slice (cell_func[approximation][nu], domain.dof_handler, 
        //                 file_name[approximation][nu], z, 0.5);
        //     };
        // };
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][z], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][z], z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{2, 0, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2x_x.gpd", z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 2, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2y_x.gpd", z, 0.5);
        puts("222222222222222222222222222222");

        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{1, 0, 0}][x][x], domain.dof_handler, 
                "stress_slice_xxx.gpd", z, 0.5);
        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{0, 1, 0}][y][x], domain.dof_handler, 
                "stress_slice_yyx.gpd", z, 0.5);

            // EPTools ::print_move<2> (slae.solution[0], domain.dof_handler, "move_approx");
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);

//         arr<u8, 4> theta  = {x, y, z, x};
//         arr<u8, 4> lambda = {x, y, z, y};
//
// #pragma omp parallel for
//         for (st n = 0; n < 4; ++n)
//         {
//             vec<arr<arr<dbl, 2>, 2>> coef_for_rhs(2);
//
//             for (auto i : {x, y})
//                 for (auto j : {x, y})
//                     for(st k = 0; k < element_matrix.C.size(); ++k)
//                     {
//                         coef_for_rhs[k][i][j] = 
//                             element_matrix.C[k][i][j][theta[n]][lambda[n]];
//                     };
//
//             slae.solution[n] = 0;
//             slae.rhsv[n] = 0;
//
//             OnCell::SourceVector<2> element_rhsv (
//                     coef_for_rhs, domain.dof_handler.get_fe());
//             OnCell::Assembler::assemble_rhsv<2> (
//                     slae.rhsv[n], element_rhsv, domain.dof_handler, bows);
//
//             dealii::SolverControl solver_control (10000, 1e-12);
//             dealii::SolverCG<> solver (solver_control);
//             solver.solve (
//                     slae.matrix,
//                     slae.solution[n],
//                     slae.rhsv[n]
//                     ,dealii::PreconditionIdentity()
//                     );
//             FOR(i, 0, slae.solution[n].size())
//                 slae.solution[n][i] = slae.solution[n][bows.subst (i)];
//         };
//
//         OnCell::SystemsLinearAlgebraicEquations<2> problem_of_torsion_rod_slae;
//         vec<ATools::SecondOrderTensor> coef_for_potr(2);
//         for (st i = 0; i < 2; ++i)
//         {
//             coef_for_potr[i][x][x] = element_matrix.C[i][x][z][x][z];
//             coef_for_potr[i][y][y] = element_matrix.C[i][y][z][y][z];
//             coef_for_potr[i][x][y] = element_matrix.C[i][x][z][y][z];
//             coef_for_potr[i][y][x] = element_matrix.C[i][x][z][y][z];
//         };
//         solve_heat_problem_on_cell_aka_torsion_rod<2> (
//                 domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);
//
//         arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
//         for (st i = 0; i < 4; ++i)
//         {
//             EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
//         };
//
//         auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
//                 domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);
//
//         for (size_t i = 0; i < 9; ++i)
//         {
//             uint8_t im = i / (dim + 1);
//             uint8_t in = i % (dim + 1);
//
//             for (size_t j = 0; j < 9; ++j)
//             {
//                 uint8_t jm = j / (dim + 1);
//                 uint8_t jn = j % (dim + 1);
//
//                 if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
//                     printf("\x1B[31m%f\x1B[0m   ", 
//                             meta_coef[im][in][jm][jn]);
//                 else
//                     printf("%f   ", 
//                             meta_coef[im][in][jm][jn]);
//             };
//             for (size_t i = 0; i < 2; ++i)
//                 printf("\n");
//         };
//         // print_tensor<6*6>(meta_coef);
//         // {
//         //     auto newcoef = unphysical_to_physicaly (meta_coef);
//         //     // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
//         //     printf("%f %f %f %f %f %f %f %f %f %f %f\n",
//         //             newcoef[0][0][0][0],
//         //             newcoef[0][0][1][1],
//         //             newcoef[0][0][2][2],
//         //             newcoef[1][1][0][0],
//         //             newcoef[1][1][1][1],
//         //             newcoef[1][1][2][2],
//         //             newcoef[2][2][0][0],
//         //             newcoef[2][2][1][1],
//         //             newcoef[2][2][2][2],
//         //             meta_coef[0][1][0][1],
//         //             meta_coef[0][2][0][2]
//         //           );
//         // };
printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa\n");
    };
    };

void solve_approx_cell_elastic_problem (cst flag, cdbl E, cdbl pua, cdbl R, cst n_p)
{
    if (flag)
    {
        enum {x, y, z};
        Domain<3> domain;
        {
            // set_cylinder(domain.grid, 0.25, y, 2);
            set_cylinder_true(domain.grid, R, z, n_p, 5);
            // set_ball(domain.grid, 0.4, 3);
            // set_rect_3d(domain.grid,
            //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
            //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
        };
        dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        OnCell::SystemsLinearAlgebraicEquations<1> slae;
        OnCell::BlackOnWhiteSubstituter bows;
        OnCell::SystemsLinearAlgebraicEquations<1> slae_2;
        OnCell::BlackOnWhiteSubstituter bows_2;

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(element_matrix.C[1]);
        EPTools ::set_isotropic_elascity{yung : 5.0, puasson : 0.22}(element_matrix.C[0]);
        // // {
        // // auto C2d = t4_to_t2 (element_matrix.C[0]);
        // //     printf("C2d\n");
        // //     for (size_t i = 0; i < 6; ++i)
        // //     {
        // //         for (size_t j = 0; j < 6; ++j)
        // //         {
        // //             if (std::abs(C2d[i][j]) > 0.0000001)
        // //                 printf("\x1B[31m%f\x1B[0m   ", 
        // //                         C2d[i][j]);
        // //             else
        // //                 printf("%f   ", 
        // //                         C2d[i][j]);
        // //         };
        // //         for (size_t i = 0; i < 2; ++i)
        // //             printf("\n");
        // //     };
        // //     printf("\n");
        // // };
        EPTools ::set_isotropic_elascity{yung : 100.0, puasson : 0.38}(element_matrix.C[1]);
        
        // {
        //     arr<arr<dbl, 6>, 6> E2d_original;
        //     // arr<arr<dbl, 6>, 6> E2d_final;
        //     for (st i = 0; i < 6; ++i)
        //     {
        //         for (st j = 0; j < 6; ++j)
        //         {
        //             E2d_original[i][j] = 0.0;
        //         };
        //     };
        //
        //     arr<dbl, 3> E = {10.0, 10.0, 10.0};
        //
        //     E2d_original[0][0] = 1.0 / E[0];
        //     E2d_original[1][1] = 1.0 / E[1];
        //     E2d_original[2][2] = 1.0 / E[2];
        //     E2d_original[0][1] = -0.25 / E[0];
        //     E2d_original[0][2] = -0.25 / E[0];
        //     E2d_original[1][0] = -0.25 / E[1];
        //     E2d_original[1][2] = -0.25 / E[1];
        //     E2d_original[2][0] = -0.25 / E[2];
        //     E2d_original[2][1] = -0.25 / E[2];
        //     E2d_original[3][3] = 1.0 / 0.4;
        //     E2d_original[4][4] = 1.0 / 0.4;
        //     E2d_original[5][5] = 1.0 / 0.4;
        //
        //     auto C2d_original = inverse (E2d_original);
        //     auto C = t2_to_t4 (C2d_original);
        //     for (st i = 0; i < 3; ++i)
        //     {
        //         for (st j = 0; j < 3; ++j)
        //         {
        //             for (st k = 0; k < 3; ++k)
        //             {
        //                 for (st l = 0; l < 3; ++l)
        //                 {
        //                     element_matrix.C[1][i][j][k][l] = C[i][j][k][l];
        //                 };
        //             };
        //         };
        //     };
        // };

        // OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);
        OnCell::prepare_system_equations_alternate<3, 3, 1> (slae, bows, domain);
        // OnCell::prepare_system_equations_alternate<3, 3, 1> (slae_2, bows_2, domain);
        // std::cout << bows.size << " " << bows_2.size << std::endl;
        // {
        //     std::ofstream f("/home/primat/tmp/bows1.txt", std::ios::out);
        //     for (st i = 0; i < bows.size; ++i)
        //     {
        //         f << bows.white[i] << " " << bows.black[i] << std::endl;
        //     };
        //     f.close ();
        // };
        // {
        //     std::ofstream f("/home/primat/tmp/bows2.txt", std::ios::out);
        //     for (st i = 0; i < bows_2.size; ++i)
        //     {
        //         f << bows_2.white[i] << " " << bows_2.black[i] << std::endl;
        //     };
        //     f.close ();
        // };
        // puts("???????????????????");
        // for (st i = 0; i < bows_2.size; ++i)
        // {
        //     std::cout << bows_2.white[i] << " " << bows_2.black[i] << std::endl;
        // };
        // for (st i = 0; i < 81; ++i)
        // {
        //     for (st j = 0; j < 81; ++j)
        //     {
        //         std::cout << slae.sparsity_pattern.exists(i,j) << "";
        //     };
        //     std::cout << std::endl;
        // };
        // std::cout << std::endl;
        // for (st i = 0; i < 81; ++i)
        // {
        //     for (st j = 0; j < 81; ++j)
        //     {
        //         std::cout << slae_2.sparsity_pattern.exists(i,j) << "";
        //     };
        //     std::cout << std::endl;
        // };
        // std::cout << std::endl;
        // {
        //     std::ofstream f("/home/primat/tmp/sp1.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // {
        //     std::ofstream f("/home/primat/tmp/sp2.txt", std::ios::out);
        //     for (st i = 0; i < domain.dof_handler.n_dofs(); ++i)
        //     {
        //         for (st j = 0; j < domain.dof_handler.n_dofs(); ++j)
        //         {
        //             f << slae_2.sparsity_pattern.exists(i,j) << "";
        //         };
        //         f << std::endl;
        //     };
        //     f.close ();
        // };
        // exit(1);

        OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

        cst number_of_approx = 2; // Начиная с нулевой
        // arr<arr<i32, 3>, number_of_approx> approximations = {
        //     arr<i32, 3>({1, 0, 0}),
        //     arr<i32, 3>({0, 1, 0})};
            // arr<i32, 3>{2, 0, 0}};
        OnCell::ArrayWithAccessToVector<arr<arr<dbl, 3>, 3>> meta_coefficient(number_of_approx+1);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> cell_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> N_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_stress (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_deform (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dbl, 3>, 3>, 3>> true_meta_coef (number_of_approx);
        printf("dfdfvdfv %d\n", slae.solution[0].size());
        for (auto &&a : meta_coefficient.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            e = 0.0;
        for (auto &&a : cell_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        printf("dfdfvdfv %d\n", cell_func.content[0][0][0][0].size());
        // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (st i = 0; i < slae.solution[0].size(); ++i)
        {
            if ((i % 3) == x) cell_func[arr<i32, 3>{0, 0, 0}][x][i] = 1.0;
            if ((i % 3) == y) cell_func[arr<i32, 3>{0, 0, 0}][y][i] = 1.0;
            if ((i % 3) == z) cell_func[arr<i32, 3>{0, 0, 0}][z][i] = 1.0;
        };
        // cell_func[arr<i32, 3>{0, 0, 0}][x][x] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][y][y] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][z][z] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (auto &&a : N_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());
        for (auto &&a : cell_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());

        // auto mean_coefficient = 
        //     OnCell::calculate_mean_coefficients<3> (domain.dof_handler, element_matrix.C);
        // auto area_of_domain = 
        //     OnCell::calculate_area_of_domain<3> (domain.dof_handler);

        OnCell::MetaCoefficientElasticCalculator mc_calculator (
                domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());

        // for (auto &&approximation : approximations)
        // {
            // auto approximation = approximations[0];
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                slae.solution[0] = 0.0;
                                slae.rhsv[0] = 0.0;
                                // printf("scsdcdfvdf %d\n", approximation);

                                OnCell::SourceVectorApprox<3> element_rhsv (approximation, nu,
                                        element_matrix.C, 
                                        meta_coefficient,
                                        cell_func,
                                        // &psi_func,
                                        domain.dof_handler.get_fe());
                                // printf("%d\n", slae.rhsv[0].size());
                                OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);

                                printf("problem %ld %ld %ld %ld\n", i, j, k, nu);
                                printf("Integ %f\n", element_rhsv.tmp);
                                dealii::SolverControl solver_control (500000, 1e-12);
                                dealii::SolverCG<> solver (solver_control);
                                solver.solve (
                                        slae.matrix,
                                        slae.solution[0],
                                        slae.rhsv[0]
                                        ,dealii::PreconditionIdentity()
                                        );
                                FOR(i, 0, slae.solution[0].size())
                                    slae.solution[0][i] = slae.solution[0][bows.subst (i)];
                                FOR(i, 0, slae.rhsv[0].size())
                                    slae.rhsv[0][i] = slae.rhsv[0][bows.subst (i)];

                                cell_func[approximation][nu] = slae.solution[0];
                                // N_func[approximation][nu] = slae.rhsv[0];
                            };
                        };
                    };
                };
            };
            puts("!!!");
            for (st i = 0; i < approx_number+2; ++i)
            {
                for (st j = 0; j < approx_number+2; ++j)
                {
                    for (st k = 0; k < approx_number+2; ++k)
                    {
                        if ((i+j+k) == approx_number+1)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                auto res = mc_calculator .calculate (
                                        approximation, nu,
                                        domain.dof_handler, cell_func);
                                meta_coefficient[approximation][nu][x] = res[x]; //E_x_a[0]_nu_a[1]
                                meta_coefficient[approximation][nu][y] = res[y]; 
                                meta_coefficient[approximation][nu][z] = res[z]; 
                                printf("meta k=(%ld, %ld, %ld) nu=%ld %f %f %f\n", i, j, k, nu, 
                                meta_coefficient[approximation][nu][x],
                                meta_coefficient[approximation][nu][y],
                                meta_coefficient[approximation][nu][z]
                                        );
                            };
                        };
                    };
                };
            };
        };
        puts("!!!!!");

        // {
        //     OnCell::StressCalculator stress_calculator (
        //             domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
        //     dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
        //     arr<i32, 3> approx = {1, 0, 0};
        //     printf("11111\n");
        //     stress_calculator .calculate (
        //             approx, x, x, x,
        //             domain.dof_handler, cell_func, stress);
        //     stress_calculator .calculate (
        //             approx, x, x, y,
        //             domain.dof_handler, cell_func, stress);
        //     // stress_calculator .calculate (
        //     //         approx, x, x, z,
        //     //         domain.dof_handler, cell_func, stress);
        //     EPTools ::print_move_slice (stress, domain.dof_handler, 
        //             "stress_slice_approx_1_0_0_x_x.gpd", z, 0.5);
        // };

        // {
        //     OnCell::StressCalculator stress_calculator (
        //             domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
        //     dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
        //     arr<i32, 3> approx = {2, 0, 0};
        //     printf("11111\n");
        //     stress_calculator .calculate (
        //             approx, x, x, x,
        //             domain.dof_handler, cell_func, stress);
        //     EPTools ::print_move_slice (stress, domain.dof_handler, 
        //             "stress_slice_approx_2x_x_x_x.gpd", z, 0.5);
        //     dbl Integrall = 0.0;
        //     for (st i = 0; i < stress.size(); ++i)
        //         Integrall += stress[i];
        //     printf("Integrall %f\n", Integrall);
        //     // {
        //     //     dealii::Vector<dbl> stress_diff_move(domain.dof_handler.n_dofs());
        //     //     for (st i = 0; i < stress.size(); ++i)
        //     //     {
        //     //         stress_diff_move[i] = (stress[i] + cell_func[arr<i32, 3>{1, 0, 0}][x][i]);
        //     //     };
        //     //     EPTools ::print_move_slice (stress_diff_move, domain.dof_handler, 
        //     //             "stress_slice_approx_2x_x_x_x_diff_move_xx.gpd", z, 0.5);
        //     // };
        // };

                printf("\n");
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
            OnCell::StressCalculatorBD stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            OnCell::DeformCalculatorBD deform_calculator (
                    domain.dof_handler, domain.dof_handler.get_fe());
            for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
            {
                for (st i = 0; i < approx_number+1; ++i)
                {
                    for (st j = 0; j < approx_number+1; ++j)
                    {
                        for (st k = 0; k < approx_number+1; ++k)
                        {
                            if ((i+j+k) == approx_number)
                            {
                                arr<i32, 3> approximation = {i, j, k};
                                for (st nu = 0; nu < 3; ++nu)
                                {
                                    for (st alpha = 0; alpha < 3; ++alpha)
                                    {
                                        dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
                                        dealii::Vector<dbl> deform(domain.dof_handler.n_dofs());
                                        for (st beta = 0; beta < 3; ++beta)
                                        {
                                            // stress_calculator .calculate (
                                            //         approximation, nu, alpha, beta,
                                            //         domain.dof_handler, cell_func, stress);
                                            stress_calculator .calculate (
                                                    approximation, nu, alpha, beta,
                                                    cell_func, stress);
                                            deform_calculator .calculate (
                                                    approximation, nu, alpha, beta,
                                                    cell_func, deform);
                                            true_meta_coef[approximation][nu][alpha][beta] =
                                                OnCell::calculate_meta_coefficients_3d_elastic_from_stress (
                                                        domain.dof_handler, stress, beta);
                                printf("meta k=(%ld, %ld, %ld) nu=%ld alpha=%ld beta=%ld %f\n", 
                                        i, j, k, nu, alpha, beta, true_meta_coef[approximation][nu][alpha][beta]);
                                        };
                                        cell_stress[approximation][nu][alpha] = stress;
                                        cell_deform[approximation][nu][alpha] = deform;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{1,0,0}][0][0], domain.dof_handler, 
                    "stress_slice_approx_x_x_x.gpd", z, 0.5);
            EPTools ::print_move_slice (cell_deform[arr<i32,3>{1,0,0}][0][0], domain.dof_handler, 
                    "deform_slice_approx_x_x_x.gpd", z, 0.5);

        // printf("Meta_xxxx %f\n", meta_coefficient[arr<i32, 3>{2, 0, 0}][x][x]);
        // printf("Meta_yxxy %f\n", meta_coefficient[arr<i32, 3>{1, 1, 0}][x][y]);
        // printf("Meta_yyxx %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][x][x]);
        // printf("Meta_xzxz %f\n", meta_coefficient[arr<i32, 3>{1, 0, 1}][x][z]);
        // printf("Meta_zzzz %f\n", meta_coefficient[arr<i32, 3>{0, 0, 2}][z][z]);
        // printf("Meta_zzyy %f\n", meta_coefficient[arr<i32, 3>{0, 1, 1}][y][z]);
        // printf("Meta_zyzy %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][z][z]);
        ATools::FourthOrderTensor meta_coef;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                meta_coef[j][x][i][x] = meta_coefficient[arr<i32, 3>{2, 0, 0}][i][j];
                meta_coef[j][x][i][y] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][x][i][z] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][y][i][x] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][y][i][y] = meta_coefficient[arr<i32, 3>{0, 2, 0}][i][j];
                meta_coef[j][y][i][z] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][x] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][z][i][y] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][z] = meta_coefficient[arr<i32, 3>{0, 0, 2}][i][j];
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        auto newcoef = unphysical_to_physicaly (meta_coef);
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
        // printf("\n");
        // auto C2d = t4_to_t2 (meta_coef);
        // auto E2d = inverse (C2d);
        // auto E = t2_to_t4 (E2d);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         1.0/E[0][0][0][0],
        //         -E[0][0][1][1]/E[0][0][0][0],
        //         -E[0][0][2][2]/E[0][0][0][0],
        //         -E[1][1][0][0]/E[1][1][1][1],
        //         1.0/E[1][1][1][1],
        //         -E[1][1][2][2]/E[1][1][1][1],
        //         -E[2][2][0][0]/E[2][2][2][2],
        //         -E[2][2][1][1]/E[2][2][2][2],
        //         1.0/E[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                for (st k = 0; k < 3; ++k)
                {
                    meta_coef[j][k][i][x] = true_meta_coef[arr<i32, 3>{1, 0, 0}][i][j][k];
                    meta_coef[j][k][i][y] = true_meta_coef[arr<i32, 3>{0, 1, 0}][i][j][k];
                    meta_coef[j][k][i][z] = true_meta_coef[arr<i32, 3>{0, 0, 1}][i][j][k];
                };
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        // auto newcoef = unphysical_to_physicaly (meta_coef);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         newcoef[0][0][0][0],
        //         newcoef[0][0][1][1],
        //         newcoef[0][0][2][2],
        //         newcoef[1][1][0][0],
        //         newcoef[1][1][1][1],
        //         newcoef[1][1][2][2],
        //         newcoef[2][2][0][0],
        //         newcoef[2][2][1][1],
        //         newcoef[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        // auto C2d = t4_to_t2 (meta_coef);
        // auto E2d = inverse (C2d);
        // auto Ef = t2_to_t4 (E2d);
        // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
        //         1.0/Ef[0][0][0][0],
        //         -Ef[0][0][1][1]/Ef[0][0][0][0],
        //         -Ef[0][0][2][2]/Ef[0][0][0][0],
        //         -Ef[1][1][0][0]/Ef[1][1][1][1],
        //         1.0/Ef[1][1][1][1],
        //         -Ef[1][1][2][2]/Ef[1][1][1][1],
        //         -Ef[2][2][0][0]/Ef[2][2][2][2],
        //         -Ef[2][2][1][1]/Ef[2][2][2][2],
        //         1.0/Ef[2][2][2][2],
        //         1.0/E2d[3][3],
        //         1.0/E2d[4][4],
        //         1.0/E2d[5][5],
        //         (C2d[0][0]-C2d[0][1])/2.0
        //         );
        // printf("\n");
        // {
        //     std::ofstream f("not_isotrop_125.gpd", std::ios::out | std::ios::app);
        //     f << E << " " << (C2d[0][0]-C2d[0][1])/2.0 << " " << 1.0/E2d[5][5] << std::endl;
        //     f.close();
        // };
        //
        //     printf("E2d_final\n");
        //     for (size_t i = 0; i < 6; ++i)
        //     {
        //         for (size_t j = 0; j < 6; ++j)
        //         {
        //             if (std::abs(E2d[i][j]) > 0.0000001)
        //                 printf("\x1B[31m%f\x1B[0m   ", 
        //                         E2d[i][j]);
        //             else
        //                 printf("%f   ", 
        //                         E2d[i][j]);
        //         };
        //         for (size_t i = 0; i < 2; ++i)
        //             printf("\n");
        //     };
        //     printf("\n");
        //
        //     printf("C2d_final\n");
        //     for (size_t i = 0; i < 6; ++i)
        //     {
        //         for (size_t j = 0; j < 6; ++j)
        //         {
        //             if (std::abs(C2d[i][j]) > 0.0000001)
        //                 printf("\x1B[31m%f\x1B[0m   ", 
        //                         C2d[i][j]);
        //             else
        //                 printf("%f   ", 
        //                         C2d[i][j]);
        //         };
        //         for (size_t i = 0; i < 2; ++i)
        //             printf("\n");
        //     };
        //     printf("\n");

        {
        std::ofstream out ("cell/meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ofstream out ("cell/stress_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/stress_"+name+".gpd", z, 0.5);
                                    };
                                    {
                                        std::ofstream out ("cell/deform_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_deform[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_deform[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/deform_"+name+".gpd", z, 0.5);
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                str name = ort[i]+str("_")+ort[j]+str("_")+ort[k]+str("_")+ort[nu];
                                {
                                    std::ofstream out ("cell/solution_"+name+".bin", std::ios::out | std::ios::binary);
                                    for (st i = 0; i < slae.solution[0].size(); ++i)
                                    {
                                        out.write ((char *) &(cell_func[approximation][nu][i]), sizeof(dbl));
                                    };
                                    out.close ();
                                };
                            };
                        };
                    };
                };
            };
        };

        {
            std::ofstream out ("cell/solution_on_cell_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution[0].size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };
            


        OnCell::ArrayWithAccessToVector<arr<str, 3>> file_name (number_of_approx);
        file_name[arr<i32, 3>{1, 0, 0}][x] = "move_slice_approx_xx.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][y] = "move_slice_approx_xy.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][z] = "move_slice_approx_xz.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][x] = "move_slice_approx_yx.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][y] = "move_slice_approx_yy.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][z] = "move_slice_approx_yz.gpd";
        EPTools ::print_coor_bin<3> (domain.dof_handler, "cell/coor_cell.bin");
    // {
    //     // puts("4444444444");
    //     printf("dofs %d\n", domain.dof_handler.n_dofs());
    //     vec<dealii::Point<dim>> coor(domain.dof_handler.n_dofs());
    //     puts("555555555");
    //     {
    //         cu8 dofs_per_cell =  domain.dof_handler.get_fe().dofs_per_cell;
    //         printf("dofs %d\n", dofs_per_cell);
    //
    //         std::vector<u32> local_dof_indices (dofs_per_cell);
    //
    //         auto cell = dof_handler.begin_active();
    //         auto endc = dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             cell ->get_dof_indices (local_dof_indices);
    //
    //             // FOR (i, 0, dofs_per_cell)
    //             //     indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
    //             FOR (i, 0, dofs_per_cell)
    //             {
    //                 coor[local_dof_indices[i]] = cell ->vertex (i);
    //             };
    //         };
    //     };
    //
    //     {
    //         std::ofstream out ("cell/coor_hole.bin", std::ios::out | std::ios::binary);
    //         for (st i = 0; i < coor.size(); ++i)
    //         {
    //             for (st j = 0; j < dim; ++j)
    //             {
    //                 out.write ((char *) &(coor[i](j)), sizeof(dbl));
    //             };
    //         };
    //         out.close ();
    //     };
    // };
        puts("222222222222222222222222222222");
        // for (auto &&approximation : approximations)
        // {
        //     for (st nu = 0; nu < 3; ++nu)
        //     {
        //         EPTools ::print_move_slice (cell_func[approximation][nu], domain.dof_handler, 
        //                 file_name[approximation][nu], z, 0.5);
        //     };
        // };
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][z], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][z], z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{2, 0, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2x_x.gpd", z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 2, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2y_x.gpd", z, 0.5);
        puts("222222222222222222222222222222");

        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{1, 0, 0}][x][x], domain.dof_handler, 
                "stress_slice_xxx.gpd", z, 0.5);
        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{0, 1, 0}][y][x], domain.dof_handler, 
                "stress_slice_yyx.gpd", z, 0.5);

            // EPTools ::print_move<2> (slae.solution[0], domain.dof_handler, "move_approx");
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);

//         arr<u8, 4> theta  = {x, y, z, x};
//         arr<u8, 4> lambda = {x, y, z, y};
//
// #pragma omp parallel for
//         for (st n = 0; n < 4; ++n)
//         {
//             vec<arr<arr<dbl, 2>, 2>> coef_for_rhs(2);
//
//             for (auto i : {x, y})
//                 for (auto j : {x, y})
//                     for(st k = 0; k < element_matrix.C.size(); ++k)
//                     {
//                         coef_for_rhs[k][i][j] = 
//                             element_matrix.C[k][i][j][theta[n]][lambda[n]];
//                     };
//
//             slae.solution[n] = 0;
//             slae.rhsv[n] = 0;
//
//             OnCell::SourceVector<2> element_rhsv (
//                     coef_for_rhs, domain.dof_handler.get_fe());
//             OnCell::Assembler::assemble_rhsv<2> (
//                     slae.rhsv[n], element_rhsv, domain.dof_handler, bows);
//
//             dealii::SolverControl solver_control (10000, 1e-12);
//             dealii::SolverCG<> solver (solver_control);
//             solver.solve (
//                     slae.matrix,
//                     slae.solution[n],
//                     slae.rhsv[n]
//                     ,dealii::PreconditionIdentity()
//                     );
//             FOR(i, 0, slae.solution[n].size())
//                 slae.solution[n][i] = slae.solution[n][bows.subst (i)];
//         };
//
//         OnCell::SystemsLinearAlgebraicEquations<2> problem_of_torsion_rod_slae;
//         vec<ATools::SecondOrderTensor> coef_for_potr(2);
//         for (st i = 0; i < 2; ++i)
//         {
//             coef_for_potr[i][x][x] = element_matrix.C[i][x][z][x][z];
//             coef_for_potr[i][y][y] = element_matrix.C[i][y][z][y][z];
//             coef_for_potr[i][x][y] = element_matrix.C[i][x][z][y][z];
//             coef_for_potr[i][y][x] = element_matrix.C[i][x][z][y][z];
//         };
//         solve_heat_problem_on_cell_aka_torsion_rod<2> (
//                 domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);
//
//         arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
//         for (st i = 0; i < 4; ++i)
//         {
//             EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
//         };
//
//         auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
//                 domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);
//
//         for (size_t i = 0; i < 9; ++i)
//         {
//             uint8_t im = i / (dim + 1);
//             uint8_t in = i % (dim + 1);
//
//             for (size_t j = 0; j < 9; ++j)
//             {
//                 uint8_t jm = j / (dim + 1);
//                 uint8_t jn = j % (dim + 1);
//
//                 if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
//                     printf("\x1B[31m%f\x1B[0m   ", 
//                             meta_coef[im][in][jm][jn]);
//                 else
//                     printf("%f   ", 
//                             meta_coef[im][in][jm][jn]);
//             };
//             for (size_t i = 0; i < 2; ++i)
//                 printf("\n");
//         };
//         // print_tensor<6*6>(meta_coef);
//         // {
//         //     auto newcoef = unphysical_to_physicaly (meta_coef);
//         //     // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
//         //     printf("%f %f %f %f %f %f %f %f %f %f %f\n",
//         //             newcoef[0][0][0][0],
//         //             newcoef[0][0][1][1],
//         //             newcoef[0][0][2][2],
//         //             newcoef[1][1][0][0],
//         //             newcoef[1][1][1][1],
//         //             newcoef[1][1][2][2],
//         //             newcoef[2][2][0][0],
//         //             newcoef[2][2][1][1],
//         //             newcoef[2][2][2][2],
//         //             meta_coef[0][1][0][1],
//         //             meta_coef[0][2][0][2]
//         //           );
//         // };
printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa\n");
    };
    };

void solve_approx_cell_elastic_problem (
        cst flag, cdbl E, cdbl pua, cdbl R, cst refine, cst number_of_approx, const str f_name)
{
    if (flag)
    {
        enum {x, y, z};

        if (access(("cell/"+f_name).c_str(), 0))
        {
            mkdir(("cell/"+f_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        };

        Domain<3> domain;
        {
            set_cylinder(domain.grid, R, y, refine);
            // set_ball(domain.grid, 0.4, 3);
            // set_rect_3d(domain.grid,
            //         dealii::Point<2>((0.5 - 0.5 / 2.0), (0.5 - 1.5 / 2.0)),
            //         dealii::Point<2>((0.5 + 0.5 / 2.0), (0.5 + 1.5 / 2.0)), 3);
        };
        dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        OnCell::SystemsLinearAlgebraicEquations<1> slae;
        OnCell::BlackOnWhiteSubstituter bows;

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
        element_matrix.C .resize (2);
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.2}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.28}(element_matrix.C[1]);
        EPTools ::set_isotropic_elascity{yung : 1.0, puasson : pua}(element_matrix.C[0]);
        // {
        // auto C2d = t4_to_t2 (element_matrix.C[0]);
        //     printf("C2d\n");
        //     for (size_t i = 0; i < 6; ++i)
        //     {
        //         for (size_t j = 0; j < 6; ++j)
        //         {
        //             if (std::abs(C2d[i][j]) > 0.0000001)
        //                 printf("\x1B[31m%f\x1B[0m   ", 
        //                         C2d[i][j]);
        //             else
        //                 printf("%f   ", 
        //                         C2d[i][j]);
        //         };
        //         for (size_t i = 0; i < 2; ++i)
        //             printf("\n");
        //     };
        //     printf("\n");
        // };
        EPTools ::set_isotropic_elascity{yung : E, puasson : 0.25}(element_matrix.C[1]);
        
        if(0)
        {
            arr<arr<dbl, 6>, 6> E2d_original;
            // arr<arr<dbl, 6>, 6> E2d_final;
            for (st i = 0; i < 6; ++i)
            {
                for (st j = 0; j < 6; ++j)
                {
                    E2d_original[i][j] = 0.0;
                };
            };

            arr<dbl, 3> E = {10.0, 100.0, 10.0};

            E2d_original[0][0] = 1.0 / E[0];
            E2d_original[1][1] = 1.0 / E[1];
            E2d_original[2][2] = 1.0 / E[2];
            E2d_original[0][1] = -0.25 / E[0];
            E2d_original[0][2] = -0.25 / E[0];
            E2d_original[1][0] = -0.25 / E[1];
            E2d_original[1][2] = -0.25 / E[1];
            E2d_original[2][0] = -0.25 / E[2];
            E2d_original[2][1] = -0.25 / E[2];
            E2d_original[3][3] = 1.0 / 0.4;
            E2d_original[4][4] = 1.0 / 0.4;
            E2d_original[5][5] = 1.0 / 0.4;

            auto C2d_original = inverse (E2d_original);
            auto C = t2_to_t4 (C2d_original);
            for (st i = 0; i < 3; ++i)
            {
                for (st j = 0; j < 3; ++j)
                {
                    for (st k = 0; k < 3; ++k)
                    {
                        for (st l = 0; l < 3; ++l)
                        {
                            element_matrix.C[1][i][j][k][l] = C[i][j][k][l];
                        };
                    };
                };
            };
        };

        OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);

        OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

        // cst number_of_approx = 3; // Начиная с нулевой
        // arr<arr<i32, 3>, number_of_approx> approximations = {
        //     arr<i32, 3>({1, 0, 0}),
        //     arr<i32, 3>({0, 1, 0})};
            // arr<i32, 3>{2, 0, 0}};
        OnCell::ArrayWithAccessToVector<arr<arr<dbl, 3>, 3>> meta_coefficient(number_of_approx+1);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> cell_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> N_func (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_stress (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_deform (number_of_approx);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dbl, 3>, 3>, 3>> true_meta_coef (number_of_approx);
        printf("dfdfvdfv %d\n", slae.solution[0].size());
        for (auto &&a : meta_coefficient.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            e = 0.0;
        for (auto &&a : cell_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        // printf("dfdfvdfv %d\n", cell_func.content[0][0][0][0].size());
        // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (st i = 0; i < slae.solution[0].size(); ++i)
        {
            if ((i % 3) == x) cell_func[arr<i32, 3>{0, 0, 0}][x][i] = 1.0;
            if ((i % 3) == y) cell_func[arr<i32, 3>{0, 0, 0}][y][i] = 1.0;
            if ((i % 3) == z) cell_func[arr<i32, 3>{0, 0, 0}][z][i] = 1.0;
        };
        // cell_func[arr<i32, 3>{0, 0, 0}][x][x] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][y][y] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        // cell_func[arr<i32, 3>{0, 0, 0}][z][z] = 1.0; // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
        for (auto &&a : N_func.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        d .reinit (slae.solution[0].size());
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());
        for (auto &&a : cell_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .reinit (slae.solution[0].size());

        // auto mean_coefficient = 
        //     OnCell::calculate_mean_coefficients<3> (domain.dof_handler, element_matrix.C);
        // auto area_of_domain = 
        //     OnCell::calculate_area_of_domain<3> (domain.dof_handler);

        OnCell::MetaCoefficientElasticCalculator mc_calculator (
                domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());

        // for (auto &&approximation : approximations)
        // {
            // auto approximation = approximations[0];
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                slae.solution[0] = 0.0;
                                slae.rhsv[0] = 0.0;
                                // printf("scsdcdfvdf %d\n", approximation);

                                OnCell::SourceVectorApprox<3> element_rhsv (approximation, nu,
                                        element_matrix.C, 
                                        meta_coefficient,
                                        cell_func,
                                        // &psi_func,
                                        domain.dof_handler.get_fe());
                                // printf("%d\n", slae.rhsv[0].size());
                                OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);

                                printf("problem %ld %ld %ld %ld\n", i, j, k, nu);
                                printf("Integ %f\n", element_rhsv.tmp);
                                dealii::SolverControl solver_control (500000, 1e-12);
                                dealii::SolverCG<> solver (solver_control);
                                solver.solve (
                                        slae.matrix,
                                        slae.solution[0],
                                        slae.rhsv[0]
                                        ,dealii::PreconditionIdentity()
                                        );
                                FOR(i, 0, slae.solution[0].size())
                                    slae.solution[0][i] = slae.solution[0][bows.subst (i)];
                                FOR(i, 0, slae.rhsv[0].size())
                                    slae.rhsv[0][i] = slae.rhsv[0][bows.subst (i)];

                                cell_func[approximation][nu] = slae.solution[0];
                                // N_func[approximation][nu] = slae.rhsv[0];
                            };
                        };
                    };
                };
            };
            puts("!!!");
            for (st i = 0; i < approx_number+2; ++i)
            {
                for (st j = 0; j < approx_number+2; ++j)
                {
                    for (st k = 0; k < approx_number+2; ++k)
                    {
                        if ((i+j+k) == approx_number+1)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                auto res = mc_calculator .calculate (
                                        approximation, nu,
                                        domain.dof_handler, cell_func);
                                meta_coefficient[approximation][nu][x] = res[x]; //E_x_a[0]_nu_a[1]
                                meta_coefficient[approximation][nu][y] = res[y]; 
                                meta_coefficient[approximation][nu][z] = res[z]; 
                                printf("meta k=(%ld, %ld, %ld) nu=%ld %f %f %f\n", i, j, k, nu, 
                                meta_coefficient[approximation][nu][x],
                                meta_coefficient[approximation][nu][y],
                                meta_coefficient[approximation][nu][z]
                                        );
                            };
                        };
                    };
                };
            };
        };
        puts("!!!!!");

        {
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
            arr<i32, 3> approx = {1, 0, 0};
            printf("11111\n");
            stress_calculator .calculate (
                    approx, x, x, x,
                    domain.dof_handler, cell_func, stress);
            stress_calculator .calculate (
                    approx, x, x, y,
                    domain.dof_handler, cell_func, stress);
            // stress_calculator .calculate (
            //         approx, x, x, z,
            //         domain.dof_handler, cell_func, stress);
            EPTools ::print_move_slice (stress, domain.dof_handler, 
                    "stress_slice_approx_1_0_0_x_x.gpd", z, 0.5);
        };

        // {
        //     OnCell::StressCalculator stress_calculator (
        //             domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
        //     dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
        //     arr<i32, 3> approx = {2, 0, 0};
        //     printf("11111\n");
        //     stress_calculator .calculate (
        //             approx, x, x, x,
        //             domain.dof_handler, cell_func, stress);
        //     EPTools ::print_move_slice (stress, domain.dof_handler, 
        //             "stress_slice_approx_2x_x_x_x.gpd", z, 0.5);
        //     dbl Integrall = 0.0;
        //     for (st i = 0; i < stress.size(); ++i)
        //         Integrall += stress[i];
        //     printf("Integrall %f\n", Integrall);
        //     // {
        //     //     dealii::Vector<dbl> stress_diff_move(domain.dof_handler.n_dofs());
        //     //     for (st i = 0; i < stress.size(); ++i)
        //     //     {
        //     //         stress_diff_move[i] = (stress[i] + cell_func[arr<i32, 3>{1, 0, 0}][x][i]);
        //     //     };
        //     //     EPTools ::print_move_slice (stress_diff_move, domain.dof_handler, 
        //     //             "stress_slice_approx_2x_x_x_x_diff_move_xx.gpd", z, 0.5);
        //     // };
        // };

                printf("\n");
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
            OnCell::StressCalculator stress_calculator (
                    domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
            for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
            {
                for (st i = 0; i < approx_number+1; ++i)
                {
                    for (st j = 0; j < approx_number+1; ++j)
                    {
                        for (st k = 0; k < approx_number+1; ++k)
                        {
                            if ((i+j+k) == approx_number)
                            {
                                arr<i32, 3> approximation = {i, j, k};
                                for (st nu = 0; nu < 3; ++nu)
                                {
                                    for (st alpha = 0; alpha < 3; ++alpha)
                                    {
                                        dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
                                        for (st beta = 0; beta < 3; ++beta)
                                        {
                                            stress_calculator .calculate (
                                                    approximation, nu, alpha, beta,
                                                    domain.dof_handler, cell_func, stress);
                                            true_meta_coef[approximation][nu][alpha][beta] =
                                                OnCell::calculate_meta_coefficients_3d_elastic_from_stress (
                                                        domain.dof_handler, stress, beta);
                                printf("meta k=(%ld, %ld, %ld) nu=%ld alpha=%ld beta=%ld %f\n", 
                                        i, j, k, nu, alpha, beta, true_meta_coef[approximation][nu][alpha][beta]);
                                        };
                                        cell_stress[approximation][nu][alpha] = stress;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{1,0,0}][0][0], domain.dof_handler, 
                    "stress_slice_approx_x_x_x.gpd", y, 0.5);
            {
            arr<str, 3> ort = {"x", "y", "z"};
            arr<str, 3> aprx = {"0", "1", "2"};
            OnCell::DeformCalculator deform_calculator (
                    domain.dof_handler, domain.dof_handler.get_fe());
            for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
            {
                for (st i = 0; i < approx_number+1; ++i)
                {
                    for (st j = 0; j < approx_number+1; ++j)
                    {
                        for (st k = 0; k < approx_number+1; ++k)
                        {
                            if ((i+j+k) == approx_number)
                            {
                                arr<i32, 3> approximation = {i, j, k};
                                for (st nu = 0; nu < 3; ++nu)
                                {
                                    dealii::Vector<dbl> deform(domain.dof_handler.n_dofs());
                                    for (st beta = 0; beta < 3; ++beta)
                                    {
                                        deform_calculator .calculate (
                                                approximation, nu, beta,
                                                domain.dof_handler, cell_func, deform);
                                        cell_deform[approximation][nu][beta] = deform;
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        // printf("Meta_xxxx %f\n", meta_coefficient[arr<i32, 3>{2, 0, 0}][x][x]);
        // printf("Meta_yxxy %f\n", meta_coefficient[arr<i32, 3>{1, 1, 0}][x][y]);
        // printf("Meta_yyxx %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][x][x]);
        // printf("Meta_xzxz %f\n", meta_coefficient[arr<i32, 3>{1, 0, 1}][x][z]);
        // printf("Meta_zzzz %f\n", meta_coefficient[arr<i32, 3>{0, 0, 2}][z][z]);
        // printf("Meta_zzyy %f\n", meta_coefficient[arr<i32, 3>{0, 1, 1}][y][z]);
        // printf("Meta_zyzy %f\n", meta_coefficient[arr<i32, 3>{0, 2, 0}][z][z]);
        ATools::FourthOrderTensor meta_coef;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                meta_coef[j][x][i][x] = meta_coefficient[arr<i32, 3>{2, 0, 0}][i][j];
                meta_coef[j][x][i][y] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][x][i][z] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][y][i][x] = meta_coefficient[arr<i32, 3>{1, 1, 0}][i][j];
                meta_coef[j][y][i][y] = meta_coefficient[arr<i32, 3>{0, 2, 0}][i][j];
                meta_coef[j][y][i][z] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][x] = meta_coefficient[arr<i32, 3>{1, 0, 1}][i][j];
                meta_coef[j][z][i][y] = meta_coefficient[arr<i32, 3>{0, 1, 1}][i][j];
                meta_coef[j][z][i][z] = meta_coefficient[arr<i32, 3>{0, 0, 2}][i][j];
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        // auto newcoef = unphysical_to_physicaly (meta_coef);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         newcoef[0][0][0][0],
        //         newcoef[0][0][1][1],
        //         newcoef[0][0][2][2],
        //         newcoef[1][1][0][0],
        //         newcoef[1][1][1][1],
        //         newcoef[1][1][2][2],
        //         newcoef[2][2][0][0],
        //         newcoef[2][2][1][1],
        //         newcoef[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        // auto C2d = t4_to_t2 (meta_coef);
        // auto E2d = inverse (C2d);
        // auto E = t2_to_t4 (E2d);
        // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
        //         1.0/E[0][0][0][0],
        //         -E[0][0][1][1]/E[0][0][0][0],
        //         -E[0][0][2][2]/E[0][0][0][0],
        //         -E[1][1][0][0]/E[1][1][1][1],
        //         1.0/E[1][1][1][1],
        //         -E[1][1][2][2]/E[1][1][1][1],
        //         -E[2][2][0][0]/E[2][2][2][2],
        //         -E[2][2][1][1]/E[2][2][2][2],
        //         1.0/E[2][2][2][2],
        //         meta_coef[0][1][0][1],
        //         meta_coef[0][2][0][2]
        //         );
        // printf("\n");
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                for (st k = 0; k < 3; ++k)
                {
                    meta_coef[j][k][i][x] = true_meta_coef[arr<i32, 3>{1, 0, 0}][i][j][k];
                    meta_coef[j][k][i][y] = true_meta_coef[arr<i32, 3>{0, 1, 0}][i][j][k];
                    meta_coef[j][k][i][z] = true_meta_coef[arr<i32, 3>{0, 0, 1}][i][j][k];
                };
            };
        };
        //
        for (size_t i = 0; i < 9; ++i)
        {
            uint8_t im = i / (2 + 1);
            uint8_t in = i % (2 + 1);

            for (size_t j = 0; j < 9; ++j)
            {
                uint8_t jm = j / (2 + 1);
                uint8_t jn = j % (2 + 1);

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
        auto newcoef = unphysical_to_physicaly (meta_coef);
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
        printf("\n");
        auto C2d = t4_to_t2 (meta_coef);
        auto E2d = inverse (C2d);
        auto Ef = t2_to_t4 (E2d);
        printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                1.0/Ef[0][0][0][0],
                -Ef[0][0][1][1]/Ef[0][0][0][0],
                -Ef[0][0][2][2]/Ef[0][0][0][0],
                -Ef[1][1][0][0]/Ef[1][1][1][1],
                1.0/Ef[1][1][1][1],
                -Ef[1][1][2][2]/Ef[1][1][1][1],
                -Ef[2][2][0][0]/Ef[2][2][2][2],
                -Ef[2][2][1][1]/Ef[2][2][2][2],
                1.0/Ef[2][2][2][2],
                1.0/E2d[3][3],
                1.0/E2d[4][4],
                1.0/E2d[5][5],
                (C2d[0][0]-C2d[0][1])/2.0
                );
        printf("\n");
        // {
        //     std::ofstream f(f_name, std::ios::out | std::ios::app);
        //     f 
        //         << E 
        //         << " " << (C2d[0][0]-C2d[0][1])/2.0 
        //         << " " << C2d[0][0] 
        //         << " " << C2d[0][1] 
        //         << " " << 1.0/E2d[5][5] 
        //         << std::endl;
        //     f.close();
        // };

            printf("E2d_final\n");
            for (size_t i = 0; i < 6; ++i)
            {
                for (size_t j = 0; j < 6; ++j)
                {
                    if (std::abs(E2d[i][j]) > 0.0000001)
                        printf("\x1B[31m%f\x1B[0m   ", 
                                E2d[i][j]);
                    else
                        printf("%f   ", 
                                E2d[i][j]);
                };
                for (size_t i = 0; i < 2; ++i)
                    printf("\n");
            };
            printf("\n");

            printf("C2d_final\n");
            for (size_t i = 0; i < 6; ++i)
            {
                for (size_t j = 0; j < 6; ++j)
                {
                    if (std::abs(C2d[i][j]) > 0.0000001)
                        printf("\x1B[31m%f\x1B[0m   ", 
                                C2d[i][j]);
                    else
                        printf("%f   ", 
                                C2d[i][j]);
                };
                for (size_t i = 0; i < 2; ++i)
                    printf("\n");
            };
            printf("\n");

        {
        std::ofstream out ("cell/"+f_name+"/meta_coef.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &meta_coef, sizeof meta_coef);
        out.close ();
        };

        {
            std::ofstream out("cell/"+f_name+"/meta_coef.txt", std::ios::out);
            for (size_t i = 0; i < 9; ++i)
            {
                uint8_t im = i / (2 + 1);
                uint8_t in = i % (2 + 1);

                for (size_t j = 0; j < 9; ++j)
                {
                    uint8_t jm = j / (2 + 1);
                    uint8_t jn = j % (2 + 1);

                    if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
                        out << "\x1B[31m" << meta_coef[im][in][jm][jn] << "[0m   ";
                    else
                        out <<  meta_coef[im][in][jm][jn] << "   ";
                };
                for (size_t i = 0; i < 2; ++i)
                    out << std::endl;
            };
            
            out.close ();
        };
        

        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ofstream out ("cell/"+f_name+"/stress_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/"+f_name+"/stress_"+name+".gpd", y, 0.5);
            EPTools ::print_move<3> (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/"+f_name+"/stress_"+name+".vtk", dealii::DataOutBase::vtk);
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ofstream out ("cell/"+f_name+"/deform_"+name+".bin", std::ios::out | std::ios::binary);
                                        for (st i = 0; i < slae.solution[0].size(); ++i)
                                        {
                                            out.write ((char *) &(cell_deform[approximation][nu][alpha][i]), sizeof(dbl));
                                        };
                                        out.close ();
            EPTools ::print_move_slice (cell_deform[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/"+f_name+"/deform_"+name+".gpd", y, 0.5);
            EPTools ::print_move<3> (cell_deform[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
                    "cell/"+f_name+"/deform_"+name+".vtk", dealii::DataOutBase::vtk);
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                str name = ort[i]+str("_")+ort[j]+str("_")+ort[k]+str("_")+ort[nu];
                                {
                                    std::ofstream out ("cell/"+f_name+"/solution_"+name+".bin", std::ios::out | std::ios::binary);
                                    for (st i = 0; i < slae.solution[0].size(); ++i)
                                    {
                                        out.write ((char *) &(cell_func[approximation][nu][i]), sizeof(dbl));
                                    };
                                    out.close ();
                                };
                            };
                        };
                    };
                };
            };
        };

        {
            std::ofstream out ("cell/"+f_name+"/solution_on_cell_size.bin", std::ios::out | std::ios::binary);
            auto size = slae.solution[0].size();
            out.write ((char *) &size, sizeof size);
            out.close ();
        };
            


        OnCell::ArrayWithAccessToVector<arr<str, 3>> file_name (number_of_approx);
        file_name[arr<i32, 3>{1, 0, 0}][x] = "move_slice_approx_xx.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][y] = "move_slice_approx_xy.gpd";
        file_name[arr<i32, 3>{1, 0, 0}][z] = "move_slice_approx_xz.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][x] = "move_slice_approx_yx.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][y] = "move_slice_approx_yy.gpd";
        file_name[arr<i32, 3>{0, 1, 0}][z] = "move_slice_approx_yz.gpd";
        EPTools ::print_coor_bin<3> (domain.dof_handler, "cell/"+f_name+"/coor_cell.bin");
    // {
    //     // puts("4444444444");
    //     printf("dofs %d\n", domain.dof_handler.n_dofs());
    //     vec<dealii::Point<dim>> coor(domain.dof_handler.n_dofs());
    //     puts("555555555");
    //     {
    //         cu8 dofs_per_cell =  domain.dof_handler.get_fe().dofs_per_cell;
    //         printf("dofs %d\n", dofs_per_cell);
    //
    //         std::vector<u32> local_dof_indices (dofs_per_cell);
    //
    //         auto cell = dof_handler.begin_active();
    //         auto endc = dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             cell ->get_dof_indices (local_dof_indices);
    //
    //             // FOR (i, 0, dofs_per_cell)
    //             //     indexes(local_dof_indices[i]) = cell ->vertex_dof_index (i, 0);
    //             FOR (i, 0, dofs_per_cell)
    //             {
    //                 coor[local_dof_indices[i]] = cell ->vertex (i);
    //             };
    //         };
    //     };
    //
    //     {
    //         std::ofstream out ("cell/coor_hole.bin", std::ios::out | std::ios::binary);
    //         for (st i = 0; i < coor.size(); ++i)
    //         {
    //             for (st j = 0; j < dim; ++j)
    //             {
    //                 out.write ((char *) &(coor[i](j)), sizeof(dbl));
    //             };
    //         };
    //         out.close ();
    //     };
    // };
        puts("222222222222222222222222222222");
        // for (auto &&approximation : approximations)
        // {
        //     for (st nu = 0; nu < 3; ++nu)
        //     {
        //         EPTools ::print_move_slice (cell_func[approximation][nu], domain.dof_handler, 
        //                 file_name[approximation][nu], z, 0.5);
        //     };
        // };
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{1, 0, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{1, 0, 0}][z], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][x], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][x], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][y], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][y], z, 0.5);
        EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 1, 0}][z], domain.dof_handler, 
                file_name[arr<i32, 3>{0, 1, 0}][z], z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{2, 0, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2x_x.gpd", z, 0.5);
        // EPTools ::print_move_slice (cell_func[arr<i32, 3>{0, 2, 0}][x], domain.dof_handler, 
        //         "move_slice_approx_2y_x.gpd", z, 0.5);
        puts("222222222222222222222222222222");

        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{1, 0, 0}][x][x], domain.dof_handler, 
                "stress_slice_xx.gpd", y, 0.5);
        EPTools ::print_move_slice (cell_stress[arr<i32, 3>{0, 1, 0}][y][x], domain.dof_handler, 
                "stress_slice_yy.gpd", y, 0.5);

        EPTools ::print_move_slice (cell_deform[arr<i32, 3>{1, 0, 0}][x][x], domain.dof_handler, 
                "deform_slice_xx.gpd", y, 0.5);
        EPTools ::print_move_slice (cell_deform[arr<i32, 3>{0, 1, 0}][y][x], domain.dof_handler, 
                "deform_slice_yy.gpd", y, 0.5);

            // EPTools ::print_move<2> (slae.solution[0], domain.dof_handler, "move_approx");
        // EPTools ::print_move_slice (slae.solution[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);
        // EPTools ::print_move_slice (slae.rhsv[0], domain.dof_handler, "move_slice_approx.gpd", z, 0.5);

//         arr<u8, 4> theta  = {x, y, z, x};
//         arr<u8, 4> lambda = {x, y, z, y};
//
// #pragma omp parallel for
//         for (st n = 0; n < 4; ++n)
//         {
//             vec<arr<arr<dbl, 2>, 2>> coef_for_rhs(2);
//
//             for (auto i : {x, y})
//                 for (auto j : {x, y})
//                     for(st k = 0; k < element_matrix.C.size(); ++k)
//                     {
//                         coef_for_rhs[k][i][j] = 
//                             element_matrix.C[k][i][j][theta[n]][lambda[n]];
//                     };
//
//             slae.solution[n] = 0;
//             slae.rhsv[n] = 0;
//
//             OnCell::SourceVector<2> element_rhsv (
//                     coef_for_rhs, domain.dof_handler.get_fe());
//             OnCell::Assembler::assemble_rhsv<2> (
//                     slae.rhsv[n], element_rhsv, domain.dof_handler, bows);
//
//             dealii::SolverControl solver_control (10000, 1e-12);
//             dealii::SolverCG<> solver (solver_control);
//             solver.solve (
//                     slae.matrix,
//                     slae.solution[n],
//                     slae.rhsv[n]
//                     ,dealii::PreconditionIdentity()
//                     );
//             FOR(i, 0, slae.solution[n].size())
//                 slae.solution[n][i] = slae.solution[n][bows.subst (i)];
//         };
//
//         OnCell::SystemsLinearAlgebraicEquations<2> problem_of_torsion_rod_slae;
//         vec<ATools::SecondOrderTensor> coef_for_potr(2);
//         for (st i = 0; i < 2; ++i)
//         {
//             coef_for_potr[i][x][x] = element_matrix.C[i][x][z][x][z];
//             coef_for_potr[i][y][y] = element_matrix.C[i][y][z][y][z];
//             coef_for_potr[i][x][y] = element_matrix.C[i][x][z][y][z];
//             coef_for_potr[i][y][x] = element_matrix.C[i][x][z][y][z];
//         };
//         solve_heat_problem_on_cell_aka_torsion_rod<2> (
//                 domain.grid, coef_for_potr, assigned_to problem_of_torsion_rod_slae);
//
//         arr<str, 4> vr = {"move_xx.gpd", "move_yy.gpd", "move_zz.gpd", "move_xy.gpd"};
//         for (st i = 0; i < 4; ++i)
//         {
//             EPTools ::print_move<2> (slae.solution[i], domain.dof_handler, vr[i]);
//         };
//
//         auto meta_coef = OnCell::calculate_meta_coefficients_2d_elastic<2> (
//                 domain.dof_handler, slae, problem_of_torsion_rod_slae, element_matrix.C);
//
//         for (size_t i = 0; i < 9; ++i)
//         {
//             uint8_t im = i / (dim + 1);
//             uint8_t in = i % (dim + 1);
//
//             for (size_t j = 0; j < 9; ++j)
//             {
//                 uint8_t jm = j / (dim + 1);
//                 uint8_t jn = j % (dim + 1);
//
//                 if (std::abs(meta_coef[im][in][jm][jn]) > 0.0000001)
//                     printf("\x1B[31m%f\x1B[0m   ", 
//                             meta_coef[im][in][jm][jn]);
//                 else
//                     printf("%f   ", 
//                             meta_coef[im][in][jm][jn]);
//             };
//             for (size_t i = 0; i < 2; ++i)
//                 printf("\n");
//         };
//         // print_tensor<6*6>(meta_coef);
//         // {
//         //     auto newcoef = unphysical_to_physicaly (meta_coef);
//         //     // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
//         //     printf("%f %f %f %f %f %f %f %f %f %f %f\n",
//         //             newcoef[0][0][0][0],
//         //             newcoef[0][0][1][1],
//         //             newcoef[0][0][2][2],
//         //             newcoef[1][1][0][0],
//         //             newcoef[1][1][1][1],
//         //             newcoef[1][1][2][2],
//         //             newcoef[2][2][0][0],
//         //             newcoef[2][2][1][1],
//         //             newcoef[2][2][2][2],
//         //             meta_coef[0][1][0][1],
//         //             meta_coef[0][2][0][2]
//         //           );
//         // };
    };
    };
void solve_two_stress (cst flag, cdbl E, cdbl pua)
{
    if (flag)
    {  
        enum {x, y, z};

        cst ort_slice = y;
        cdbl coor_slice = 0.5;

        ATools::FourthOrderTensor C;
        std::ifstream in ("cell/meta_coef.bin", std::ios::in | std::ios::binary);
        in.read ((char *) &C, sizeof C);

        st size_sol_hole = 0;
        {
            std::ifstream in ("hole/solution_hole_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_hole, sizeof size_sol_hole);
            in.close ();
        };
        st size_sol_cell = 0;
        {
            std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_cell, sizeof size_sol_cell);
            in.close ();
        };
        arr<vec<dbl>, 2> deform_1;
        arr<arr<vec<dbl>, 2>, 2> deform_2;
        deform_1[x] .resize (size_sol_hole);
        deform_1[y] .resize (size_sol_hole);
        deform_2[x][x] .resize (size_sol_hole);
        deform_2[x][y] .resize (size_sol_hole);
        deform_2[y][x] .resize (size_sol_hole);
        deform_2[y][y] .resize (size_sol_hole);
        {
            std::ifstream in ("hole/deform_hole_x.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_y.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/stress_hole_y.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_xx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_xy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_yx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/deform_hole_yy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][y][i], sizeof(dbl));
            };
            in.close ();
        };

        cst number_of_approx = 3;

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_stress (number_of_approx);
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        // std::cout << "cell/stress_"+name+".bin" << std::endl;
                                        std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::cout << in.is_open() << std::endl;
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_stress[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        // {
        //     // std::cout << "cell/stress_"+name+".bin" << std::endl;
        //     std::ifstream in ("cell/stress_1_0_0_x_x.bin", std::ios::in | std::ios::binary);
        //     // std::cout << in.is_open() << std::endl;
        //     arr<dbl, 3> tmp;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
        //         dbl tmp = 0.0;
        //         in.read ((char *) &(tmp), sizeof(dbl));
        //         cell_stress[arr<i32, 3>{1, 0, 0}][x][x][i] = tmp;
        //     };
        //     in.close ();
        // };

        vec<dealii::Point<2>> coor_hole (size_sol_hole);
        {
            std::ifstream in ("hole/coor_hole.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &coor_hole[i](x), sizeof(dbl));
                in.read ((char *) &coor_hole[i](y), sizeof(dbl));
            };
            in.close ();
        };

        vec<dealii::Point<3>> coor_cell (size_sol_cell);
        {
            std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_cell; ++i)
            {
                in.read ((char *) &coor_cell[i](x), sizeof(dbl));
                in.read ((char *) &coor_cell[i](y), sizeof(dbl));
                in.read ((char *) &coor_cell[i](z), sizeof(dbl));
            };
            in.close ();
        };
        {
            FILE *F;
            F = fopen("stress_cell_yyy.gpd", "w");
        for (st i = 0; i < size_sol_cell; ++i)
        {
            if (i % 3 == y)
            fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
        };
            fclose(F);
        };
        // for (st i = 0; i < 10; ++i)
        // {
        //     printf("%f %f %f\n", coor_cell[i](x), coor_cell[i](y), coor_cell[i](z));
        // };
        
        st size_line_hole = 0;
        {
            FILE *F;
            F = fopen("deform_1_yy_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == y)
                    {
                fprintf(F,"%f %f\n", coor_hole[i](x), deform_1[y][i]);
                // fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[x][y][i]);
                ++size_line_hole;
                // printf("%f\n", deform_1[y][i]);
                    };
                };
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_1_yyx_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == x)
                    {
                        fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[y][y][i]);
                    };
                };
            };
            fclose(F);
        };
        st size_line_cell = 0;
        {
            FILE *F;
            F = fopen("stress_cell_line.gpd", "w");
            for (st i = 0; i < size_sol_cell; ++i)
            {
                if (
                        (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10) and
                        (std::abs(coor_cell[i](z) - 0.5) < 1.0e-10)
                   )

                {
                    if ((i % 3) == y)
                    {
                fprintf(F,"%f %f\n", coor_cell[i](x), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
                ++size_line_cell;
                    };
                };
            };
            fclose(F);
        };

        arr<arr<vec<dbl>, 2>, 2> deform_line_1;
        arr<arr<arr<vec<dbl>, 2>, 2>, 2> deform_line_2;
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               deform_line_1[i][j] .resize (size_line_hole);
               
           }; 
        };
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               for (st k = 0; k < 2; ++k)
               {
               deform_line_2[i][j][k] .resize (size_line_hole);
               };
           }; 
        };
        vec<dealii::Point<2>> coor_line_hole (size_line_hole);
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            if ((i % 2) == k)
                            {
                                deform_line_1[k][j][n] = deform_1[j][i];
                                // printf("%d\n", n);
                            };
                        };
                    };
                    if ((i % 2))
                    {
                coor_line_hole[n] = coor_hole[i];
                ++n;
                    };
                };
            };
        };
        {
            FILE *F;
            F = fopen("deform_1_yy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_1[y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_1_xx_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_1[x][x][i]);
            };
            fclose(F);
        };
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            for (st l = 0; l < 2; ++l)
                            {
                                if ((i % 2) == k)
                                {
                                    deform_line_2[k][j][l][n] = deform_2[j][l][i];
                                };
                            };
                        };
                    };
                    if ((i % 2))
                    {
                ++n;
                    };
                };
            };
        };
        {
            FILE *F;
            F = fopen("deform_2_yyx_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](x), deform_line_2[y][y][x][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_2_yyy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_2[y][y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_2_xxy_line_2.gpd", "w");
            for (st i = 0; i < size_line_hole; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_hole[i](y), deform_line_2[x][x][y][i]);
            };
            fclose(F);
        };
        //     // for (st i = 0; i < size_line_hole; ++i)
        //     // {
        //     //     printf("%f %f\n", coor_line_hole[i](x), coor_line_hole[i](y));
        //     // };
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_line_stress (number_of_approx);
        for (auto &&a : cell_line_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_line_cell);
        vec<dealii::Point<3>> coor_line_cell (size_line_cell);
        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10) and
                        (std::abs(coor_cell[m](z) - 0.5) < 1.0e-10)
                   )

                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                        cell_line_stress[approximation][nu][alpha][beta][n] = 
                                                             cell_stress[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    if ((m % 3) == z)
                    {
                        coor_line_cell[n] = coor_cell[m];
                        ++n;
                    };
                    if (n == size_line_cell)
                        break;
                };
            };
        };
        // {
        //     st n = 0;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (
        //                 (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10) and
        //                 (std::abs(coor_cell[i](z) - 0.5) < 1.0e-10)
        //            )
        //
        //         {
        //             st beta = y;
        //             {
        //                 if ((i % 3) == beta)
        //                 {
        //                     cell_line_stress[arr<i32, 3>{0,1,0}][y][y][y][n] = 
        //                          cell_stress[arr<i32, 3>{0,1,0}][y][y][i];
        //                 ++n;
        //                 };
        //             };
        //         };
        //     };
        // };
        // {
        //     FILE *F;
        //     F = fopen("stress_cell_flat.gpd", "w");
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (
        //                 (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10)
        //            )
        //
        //         {
        //             if ((i % 3) == 0)
        //         fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{1,0,0}][x][x][i]);
        //         };
        //     };
        //     fclose(F);
        // };
        {
            FILE *F;
            F = fopen("stress_cell_line_2.gpd", "w");
            for (st i = 0; i < size_line_cell; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][i]);
                // printf("%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][i]);
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("stress_cell_line_2.gpd", "w");
            for (st i = 0; i < size_line_cell; ++i)
            {
                fprintf(F,"%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][i]);
                // printf("%f %f\n", coor_line_cell[i](x), cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][i]);
            };
            fclose(F);
        };

        printf("size_line %d %d\n", size_line_hole, size_line_cell);
        printf("size %d %d\n", size_sol_hole, size_sol_cell);

        cst num_cells = 10;
        cdbl cell_size = 1.0 / num_cells;
        for (st i = 0; i < size_line_cell; ++i)
        {
            coor_line_cell[i](x) /= num_cells;
        };

        dbl max_macro_stress = 0.0;
        dbl max_final_stress = 0.0;

        cst fgh = 10000;
        arr<arr<vec<dbl>, 3>, 3> macro_stress;
        arr<arr<vec<dbl>, 3>, 3> final_stress;
        arr<arr<vec<dbl>, 3>, 3> final_stress_2;
        for (st i = 0; i < 3; ++i)
        {
           for (st j = 0; j < 3; ++j)
           {
               // macro_stress[i][j] .resize (size_line_hole);
               // final_stress[i][j] .resize (size_line_hole);
               // final_stress_2[i][j] .resize (size_line_hole);
               macro_stress[i][j] .resize (fgh);
               final_stress[i][j] .resize (fgh);
               final_stress_2[i][j] .resize (fgh);
           }; 
        };
        {
            FILE *F;
            F = fopen("hole_plas_cell.gpd", "w");
            // for (st i = 0; i < size_line_hole; ++i)
            for (st i = 0; i < fgh+1; ++i)
            {
                // dbl coor_in_cell = coor_line_hole[i](x);
                dbl coor_in_hole = 1.0 / fgh * i;
                dbl coor_in_cell = coor_in_hole;
                for (st j = 0; j < num_cells; ++j)
                {
                    // printf("%f\n", coor_in_cell);
                    coor_in_cell -= cell_size;
                    if (coor_in_cell < 0.0)
                    {
                        coor_in_cell += cell_size;
                        break;
                    };
                };
                // printf("%f %f %f\n", 
                //         1.0 / fgh * i,
                //         coor_in_cell, cell_size);
                // printf("\n");
                // dbl sol_in_cell = 0.0;
                st num_of_point_in_cell = 0;
                for (st j = 0; j < size_line_cell; ++j)
                {
                    if (coor_in_cell < coor_line_cell[j](x))
                    {
                        // dbl X = coor_in_cell;
                        // dbl X1 = coor_line_cell[j-1](x);
                        // dbl X2 = coor_line_cell[j](x);
                        // dbl Y1 = cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][j-1];
                        // dbl Y2 = cell_line_stress[arr<i32,3>{1,0,0}][x][x][x][j];
                        // sol_in_cell = (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        num_of_point_in_cell = j;
                        // printf("%f %f %f\n", coor_line_hole[i](x), coor_in_cell, coor_line_cell[j](x));
                        break;
                    };
                };
                st num_of_point_in_hole = 0;
                for (st j = 0; j < size_line_hole; ++j)
                {
                    if ((coor_in_hole) < coor_line_hole[j](x))
                    {
                        num_of_point_in_hole = j;
                        break;
                    };
                };
                auto sol_in_cell  = [&coor_in_cell, &coor_line_cell, &cell_line_stress, num_of_point_in_cell] 
                    (cst i, cst j, cst k, cst l, cst m, cst n)
                    {
                        cst nm = num_of_point_in_cell;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_cell.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_cell;
                        cdbl X1 = coor_line_cell[nm_1](0);
                        cdbl X2 = coor_line_cell[nm](0);
                        cdbl Y1 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm_1];
                        cdbl Y2 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm];
                        // printf("%f %f %f %f %f %f %ld %ld %f\n",
                        //         (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2), X, X1, X2, Y1, Y2, nm_1, nm,
                        //         cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][0]);
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                auto sol_in_hole  = [&coor_in_hole, &coor_line_hole, &deform_line_1, num_of_point_in_hole] 
                    (cst i, cst j)
                    {
                        cst nm = num_of_point_in_hole;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_hole.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_hole;
                        cdbl X1 = coor_line_hole[nm_1](0);
                        cdbl X2 = coor_line_hole[nm](0);
                        cdbl Y1 = deform_line_1[i][j][nm_1];
                        cdbl Y2 = deform_line_1[i][j][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                // cst num = num_of_point_in_cell;

                for (st alpha = 0; alpha < 3; ++alpha)
                {
                    for (st beta = 0; beta < 3; ++beta)
                    {
                       macro_stress[alpha][beta][i] = 
                           C[x][x][alpha][beta] * sol_in_hole(x,x) +
                           C[y][x][alpha][beta] * sol_in_hole(y,x) +
                           C[x][y][alpha][beta] * sol_in_hole(x,y) +
                           C[y][y][alpha][beta] * sol_in_hole(y,y);
                       final_stress[alpha][beta][i] = 
                           sol_in_cell(1,0,0,x,alpha,beta) * sol_in_hole(x,x) +
                           sol_in_cell(1,0,0,y,alpha,beta) * sol_in_hole(y,x) +
                           sol_in_cell(0,1,0,x,alpha,beta) * sol_in_hole(x,y) +
                           sol_in_cell(0,1,0,y,alpha,beta) * sol_in_hole(y,y);
                    };
                };
                // for (st alpha = 0; alpha < 3; ++alpha)
                // {
                //     for (st beta = 0; beta < 3; ++beta)
                //     {
                //        macro_stress[alpha][beta][i] = 
                //            C[x][x][alpha][beta] * deform_line_1[x][x][i] +
                //            C[y][x][alpha][beta] * deform_line_1[y][x][i] +
                //            C[x][y][alpha][beta] * deform_line_1[x][y][i] +
                //            C[y][y][alpha][beta] * deform_line_1[y][y][i];
                //        final_stress[alpha][beta][i] = 
                //            sol_in_cell(1,0,0,x,alpha,beta, num) * deform_line_1[x][x][i] +
                //            sol_in_cell(1,0,0,y,alpha,beta, num) * deform_line_1[y][x][i] +
                //            sol_in_cell(0,1,0,x,alpha,beta, num) * deform_line_1[x][y][i] +
                //            sol_in_cell(0,1,0,y,alpha,beta, num) * deform_line_1[y][y][i];
                //        final_stress_2[alpha][beta][i] = 
                //            sol_in_cell(1,0,0,x,alpha,beta, num) * deform_line_1[x][x][i] +
                //            sol_in_cell(1,0,0,y,alpha,beta, num) * deform_line_1[y][x][i] +
                //            sol_in_cell(0,1,0,x,alpha,beta, num) * deform_line_1[x][y][i] +
                //            sol_in_cell(0,1,0,y,alpha,beta, num) * deform_line_1[y][y][i] +
                //            (
                //            sol_in_cell(2,0,0,x,alpha,beta, num) * deform_line_2[x][x][x][i] +
                //            sol_in_cell(1,1,0,x,alpha,beta, num) * deform_line_2[x][y][x][i] +
                //            sol_in_cell(1,1,0,x,alpha,beta, num) * deform_line_2[x][x][y][i] +
                //            sol_in_cell(0,2,0,x,alpha,beta, num) * deform_line_2[x][y][y][i] +
                //            sol_in_cell(2,0,0,y,alpha,beta, num) * deform_line_2[y][x][x][i] +
                //            sol_in_cell(1,1,0,y,alpha,beta, num) * deform_line_2[y][y][x][i] +
                //            sol_in_cell(1,1,0,y,alpha,beta, num) * deform_line_2[y][x][y][i] +
                //            sol_in_cell(0,2,0,y,alpha,beta, num) * deform_line_2[y][y][y][i]
                //            ) * cell_size;
                //     };
                // };
                //
                fprintf(F, "%f %f %f %f %f\n", 
                        coor_in_hole, 
                        sol_in_cell(0,1,0,y,y,y), sol_in_hole(y,y), 
                        final_stress[y][y][i], macro_stress[y][y][i]);
                for (st i = 0; i < fgh; ++i)
                {
                    if ((coor_in_hole > cell_size * (43-1)) and (coor_in_hole < cell_size * (43)))
                    // if ((coor_in_hole > cell_size * (22-1)) and (coor_in_hole < cell_size * (22)))
                            {
                    if (max_macro_stress < macro_stress[y][y][i])
                        max_macro_stress = macro_stress[y][y][i];
                    if (max_final_stress < final_stress[y][y][i])
                        max_final_stress = final_stress[y][y][i];
                        };
                };
                // fprintf(F, "%f %f %f %f %f %f %f %f\n", 
                //         coor_line_hole[i](x), 
                //         deform_line_1[y][y][i], deform_line_2[y][y][x][i],
                //         sol_in_cell(0,1,0,y,y,y,num), sol_in_cell(0,2,0,y,x,x,num), 
                //         final_stress[y][y][i], final_stress_2[y][y][i], macro_stress[y][y][i]);
                // fprintf(F, "%f %f %f\n", 
                //         coor_in_hole,
                //         coor_in_cell,
                //         sol_in_cell(0,1,0,y,y,y));
                // fprintf(F, "%f %f %f %f %f %f %f\n", 
                //         coor_line_hole[i](y), deform_line_1[x][x][i], deform_line_2[x][x][y][i],
                //         sol_in_cell(1,0,0,x,x,x,num), sol_in_cell(2,0,0,x,y,y,num), 
                //         final_stress[x][x][i], final_stress_2[x][x][i]);
        // for (st o = 0; o < size_line_cell; ++o)
        // {
        //     printf("%f %f %f\n", cell_line_stress[arr<i32,3>{0,1,0}][y][y][y][o],
        //             sol_in_cell(0,1,0,y,y,y,o), coor_in_cell);
        // };
        // puts(" ");
            };
            fclose(F);
        };
        // dbl max_macro_stress = 0.0;
        // dbl max_final_stress = 0.0;
        // for (st i = 0; i < fgh; ++i)
        // {
        //     // if (
        //     if (max_macro_stress < macro_stress[y][y][i])
        //         max_macro_stress = macro_stress[y][y][i];
        //     if (max_final_stress < final_stress[y][y][i])
        //         max_final_stress = final_stress[y][y][i];
        // };
        {
            FILE *F;
            F = fopen("pua_25_2.gpd", "a");
            fprintf(F, "%f %f %f\n", pua, max_macro_stress, max_final_stress);
            fclose(F);
        };
    };
};

void get_line_deform_and_stress(
        cst ort_slice,
        cdbl coor_slice,
        ATools::FourthOrderTensor &C,
        arr<arr<vec<dbl>, 2>, 2> &deform_line_1,
        arr<arr<arr<vec<dbl>, 2>, 2>, 2> &deform_line_2,
        vec<dealii::Point<2>> &coor_line_hole,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> &cell_line_stress,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> &cell_line_deform,
        vec<dealii::Point<3>> &coor_line_cell,
        cst number_of_approx,
        const str f_name_cell, const str f_name_hole
        )
{
        enum {x, y, z};

        ///////////////////////////////// Техническая часть, добывание из файлов, сформированных решением
        ///////////////////////////////// двух задач выше и формирование линии вдоль которой будут 
        ///////////////////////////////// вычислятся напряжения в дальнейшем.

        std::ifstream in ("cell/"+f_name_cell+"/meta_coef.bin", std::ios::in | std::ios::binary);
        // std::ifstream in ("cell/meta_coef.bin", std::ios::in | std::ios::binary);
        in.read ((char *) &C, sizeof C);

        st size_sol_hole = 0;
        {
            std::ifstream in ("hole/"+f_name_hole+"/solution_hole_size.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/solution_hole_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_hole, sizeof size_sol_hole);
            in.close ();
        };
        st size_sol_cell = 0;
        {
            std::ifstream in ("cell/"+f_name_cell+"/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_cell, sizeof size_sol_cell);
            in.close ();
        };
        arr<vec<dbl>, 2> deform_1;
        arr<arr<vec<dbl>, 2>, 2> deform_2;
        deform_1[x] .resize (size_sol_hole);
        deform_1[y] .resize (size_sol_hole);
        deform_2[x][x] .resize (size_sol_hole);
        deform_2[x][y] .resize (size_sol_hole);
        deform_2[y][x] .resize (size_sol_hole);
        deform_2[y][y] .resize (size_sol_hole);
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_x.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_x.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_y.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_y.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/stress_hole_y.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_1[y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_xx.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_xx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_xy.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_xy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[x][y][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_yx.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_yx.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][x][i], sizeof(dbl));
            };
            in.close ();
        };
        {
            std::ifstream in ("hole/"+f_name_hole+"/deform_hole_yy.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/deform_hole_yy.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &deform_2[y][y][i], sizeof(dbl));
            };
            in.close ();
        };

        // cst number_of_approx = 3;

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_stress (number_of_approx);
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_deform (number_of_approx);
        for (auto &&a : cell_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        // std::cout << "cell/stress_"+name+".bin" << std::endl;
                                        std::ifstream in ("cell/"+f_name_cell+"/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::cout << in.is_open() << std::endl;
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_stress[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ifstream in (
                                                "cell/"+f_name_cell+
                                                "/deform_"+name+".bin", std::ios::in | std::ios::binary);
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_deform[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        // {
        //     // std::cout << "cell/stress_"+name+".bin" << std::endl;
        //     std::ifstream in ("cell/stress_1_0_0_x_x.bin", std::ios::in | std::ios::binary);
        //     // std::cout << in.is_open() << std::endl;
        //     arr<dbl, 3> tmp;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
        //         dbl tmp = 0.0;
        //         in.read ((char *) &(tmp), sizeof(dbl));
        //         cell_stress[arr<i32, 3>{1, 0, 0}][x][x][i] = tmp;
        //     };
        //     in.close ();
        // };

        vec<dealii::Point<2>> coor_hole (size_sol_hole);
        {
            std::ifstream in ("hole/"+f_name_hole+"/coor_hole.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("hole/coor_hole.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_hole; ++i)
            {
                in.read ((char *) &coor_hole[i](x), sizeof(dbl));
                in.read ((char *) &coor_hole[i](y), sizeof(dbl));
            };
            in.close ();
        };

        vec<dealii::Point<3>> coor_cell (size_sol_cell);
        {
            std::ifstream in ("cell/"+f_name_cell+"/coor_cell.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_cell; ++i)
            {
                in.read ((char *) &coor_cell[i](x), sizeof(dbl));
                in.read ((char *) &coor_cell[i](y), sizeof(dbl));
                in.read ((char *) &coor_cell[i](z), sizeof(dbl));
            };
            in.close ();
        };
        {
            FILE *F;
            F = fopen("stress_cell_yyy.gpd", "w");
        for (st i = 0; i < size_sol_cell; ++i)
        {
            if (i % 3 == y)
            fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
        };
            fclose(F);
        };
        // for (st i = 0; i < 10; ++i)
        // {
        //     printf("%f %f %f\n", coor_cell[i](x), coor_cell[i](y), coor_cell[i](z));
        // };
        
        st size_line_hole = 0;
        {
            FILE *F;
            F = fopen("deform_1_yy_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == y)
                    {
                fprintf(F,"%f %f\n", coor_hole[i](x), deform_1[y][i]);
                // fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[x][y][i]);
                ++size_line_hole;
                // printf("%f\n", deform_1[y][i]);
                    };
                };
            };
            fclose(F);
        };
        {
            FILE *F;
            F = fopen("deform_1_yyx_line.gpd", "w");
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](y) - 0.5) < 1.0e-10)
                {
                    if ((i % 2) == x)
                    {
                        fprintf(F,"%f %f\n", coor_hole[i](x), deform_2[y][y][i]);
                    };
                };
            };
            fclose(F);
        };
        st size_line_cell = 0;
        {
            FILE *F;
            F = fopen("stress_cell_line.gpd", "w");
            for (st i = 0; i < size_sol_cell; ++i)
            {
                if (
                        (std::abs(coor_cell[i](y) - 0.5) < 1.0e-10) and
                        (std::abs(coor_cell[i](z) - 0.5) < 1.0e-10)
                   )

                {
                    if ((i % 3) == y)
                    {
                fprintf(F,"%f %f\n", coor_cell[i](x), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
                ++size_line_cell;
                    };
                };
            };
            fclose(F);
        };

        // To line hole
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               deform_line_1[i][j] .resize (size_line_hole);
               
           }; 
        };
        for (st i = 0; i < 2; ++i)
        {
           for (st j = 0; j < 2; ++j)
           {
               for (st k = 0; k < 2; ++k)
               {
               deform_line_2[i][j][k] .resize (size_line_hole);
               };
           }; 
        };
        coor_line_hole .resize(size_line_hole);
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            if ((i % 2) == k)
                            {
                                deform_line_1[k][j][n] = deform_1[j][i];
                                // printf("%d\n", n);
                            };
                        };
                    };
                    if ((i % 2))
                    {
                coor_line_hole[n] = coor_hole[i];
                ++n;
                    };
                };
            };
        };
        {
            st n = 0;
            for (st i = 0; i < size_sol_hole; ++i)
            {
                if (std::abs(coor_hole[i](ort_slice) - coor_slice) < 1.0e-10)
                {
                    for (st j = 0; j < 2; ++j)
                    {
                        for (st k = 0; k < 2; ++k)
                        {
                            for (st l = 0; l < 2; ++l)
                            {
                                if ((i % 2) == k)
                                {
                                    deform_line_2[k][j][l][n] = deform_2[j][l][i];
                                };
                            };
                        };
                    };
                    if ((i % 2))
                    {
                ++n;
                    };
                };
            };
        };

        // to line stress
        for (auto &&a : cell_line_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_line_cell);
        coor_line_cell .resize(size_line_cell);
        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10) and
                        (std::abs(coor_cell[m](z) - 0.5) < 1.0e-10)
                   )

                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                        cell_line_stress[approximation][nu][alpha][beta][n] = 
                                                             cell_stress[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    if ((m % 3) == z)
                    {
                        coor_line_cell[n] = coor_cell[m];
                        ++n;
                    };
                    if (n == size_line_cell)
                        break;
                };
            };
        };

        // to line deform
        for (auto &&a : cell_line_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_line_cell);
        coor_line_cell .resize(size_line_cell);
        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10) and
                        (std::abs(coor_cell[m](z) - 0.5) < 1.0e-10)
                   )

                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                        cell_line_deform[approximation][nu][alpha][beta][n] = 
                                                             cell_deform[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    if ((m % 3) == z)
                    {
                        coor_line_cell[n] = coor_cell[m];
                        ++n;
                    };
                    if (n == size_line_cell)
                        break;
                };
            };
        };

        printf("size_line %d %d\n", size_line_hole, size_line_cell);
        printf("size %d %d\n", size_sol_hole, size_sol_cell);
    
};

void get_flat_deform_and_stress(
        cst ort_slice,
        cdbl coor_slice,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> &cell_flat_stress,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> &cell_flat_deform,
        vec<dealii::Point<3>> &coor_flat_cell,
        cst number_of_approx
        )
{
        enum {x, y, z};

        ///////////////////////////////// Техническая часть, добывание из файлов, сформированных решением
        ///////////////////////////////// двух задач выше и формирование линии вдоль которой будут 
        ///////////////////////////////// вычислятся напряжения в дальнейшем.

        st size_sol_cell = 0;
        {
            std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_cell, sizeof size_sol_cell);
            in.close ();
        };

        // cst number_of_approx = 3;

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_stress (number_of_approx);
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_deform (number_of_approx);
        for (auto &&a : cell_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        approx_iteration (number_of_approx, [&cell_stress, size_sol_cell](arr<i32, 3> a, cst nu, cst alpha){
                arr<str, 3> ort = {"x", "y", "z"};
                arr<str, 3> aprx = {"0", "1", "2"};
                str name = aprx[a[0]]+str("_")+aprx[a[1]]+str("_")+aprx[a[2]]+str("_")+ort[nu]+str("_")+ort[alpha];
                {
                std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                arr<dbl, 3> tmp;
                for (st i = 0; i < size_sol_cell; ++i)
                {
                dbl tmp = 0.0;
                in.read ((char *) &(tmp), sizeof(dbl));
                cell_stress[a][nu][alpha][i] = tmp;
                };
                in.close ();
                };
                });

        approx_iteration (number_of_approx, [&cell_deform, size_sol_cell](arr<i32, 3> a, cst nu, cst alpha){
                arr<str, 3> ort = {"x", "y", "z"};
                arr<str, 3> aprx = {"0", "1", "2"};
                str name = aprx[a[0]]+str("_")+aprx[a[1]]+str("_")+aprx[a[2]]+str("_")+ort[nu]+str("_")+ort[alpha];
                {
                std::ifstream in ("cell/deform_"+name+".bin", std::ios::in | std::ios::binary);
                arr<dbl, 3> tmp;
                for (st i = 0; i < size_sol_cell; ++i)
                {
                dbl tmp = 0.0;
                in.read ((char *) &(tmp), sizeof(dbl));
                cell_deform[a][nu][alpha][i] = tmp;
                };
                in.close ();
                };
                });

        vec<dealii::Point<3>> coor_cell (size_sol_cell);
        {
            std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_cell; ++i)
            {
                in.read ((char *) &coor_cell[i](x), sizeof(dbl));
                in.read ((char *) &coor_cell[i](y), sizeof(dbl));
                in.read ((char *) &coor_cell[i](z), sizeof(dbl));
            };
            in.close ();
        };
        puts("olololo");
        std::cout << coor_cell[0](z) << " " << coor_cell[1](z) << " " << coor_cell[2](z) << std::endl;

        for (auto &&a : cell_flat_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_sol_cell / 15);

        for (auto &&a : cell_flat_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                                j .resize (size_sol_cell / 15);

        coor_flat_cell .resize(size_sol_cell / 15);

        std::cout << coor_flat_cell.size() << std::endl;
        {
            cst coort_1 = (ort_slice + 1) % 3; 
            cst coort_2 = (ort_slice + 2) % 3;
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10) and
                        (m % 3 == 0)
                   )
                {
                    coor_flat_cell[n] = coor_cell[m];
                    ++n;
                };
            };
            std::cout << size_sol_cell /5 << " " << n << std::endl;
        };
        {
            std::ofstream f("coor.gpd", std::ios::out);
            for (st i = 0; i < size_sol_cell / 15; ++i)
                f << coor_flat_cell[i](x) << " " << coor_flat_cell[i](y) << " " << 
                    i % 1 << std::endl;
            f.close ();
        };

        approx_iteration (number_of_approx, [&cell_stress, &cell_flat_stress, &coor_cell, size_sol_cell, ort_slice, coor_slice](arr<i32, 3> a, cst nu, cst alpha){
                st n = 0;
                for (st m = 0; m < size_sol_cell; ++m)
                {
                if (
                    (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10)
                   )
                {
                st beta = m % 3;
                cell_flat_stress[a][nu][alpha][beta][n] = cell_stress[a][nu][alpha][m];
                if (beta == 2)
                {
                ++n;
                };
                };
                };
                // for (st m = 0; m < size_sol_cell / 3; ++m)
                // {
                // cst n = m*3;
                // if (
                //         (std::abs(coor_cell[n](ort_slice) - coor_slice) < 1.0e-10)
                //    )
                // {
                //     cell_flat_stress[a][nu][alpha][0][0] = 1.0;//cell_stress[a][nu][alpha][n];
                //     // cell_flat_stress[a][nu][alpha][1][m] = cell_stress[a][nu][alpha][n+1];
                //     // cell_flat_stress[a][nu][alpha][1][m] = cell_stress[a][nu][alpha][n+2];
                // };
                // };
        });

        approx_iteration (number_of_approx, [&cell_deform, &cell_flat_deform, &coor_cell, size_sol_cell, ort_slice, coor_slice](arr<i32, 3> a, cst nu, cst alpha){
                st n = 0;
                for (st m = 0; m < size_sol_cell; ++m)
                {
                if (
                    (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10)
                   )
                {
                st beta = m % 3;
                cell_flat_deform[a][nu][alpha][beta][n] = 
                cell_deform[a][nu][alpha][m];
                if (beta == 2)
                {
                ++n;
                };
                };
                };
                });

        // {
        //     std::ofstream f("stress.gpd", std::ios::out);
        //     for (st i = 0; i < size_sol_cell / 15; ++i)
        //         f << coor_flat_cell[i](x) << 
        //             " " << coor_flat_cell[i](y) << 
        //             " " << 
        //             // 0.0
        //             cell_flat_stress[arr<i32, 3>{1,0,0}][0][0][1][i] 
        //             << std::endl;
        //     f.close ();
        // };
        //
        // {
        //     std::ofstream f("deform.gpd", std::ios::out);
        //     for (st i = 0; i < size_sol_cell / 15; ++i)
        //         f << coor_flat_cell[i](x) << 
        //             " " << coor_flat_cell[i](y) << 
        //             " " << 
        //             // 0.0
        //             cell_flat_deform[arr<i32, 3>{1,0,0}][0][0][0][i] 
        //             << std::endl;
        //     f.close ();
        // };
};

void get_flat_deform_stress_and_domain (
        cst ort_slice,
        cdbl coor_slice,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &cell_stress,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &cell_deform,
        Domain<3> domain_cell,
        cst number_of_approx
        )
{
    // OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_stress(3);
    // OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_deform(3);
    // vec<dealii::Point<3>> coor_flat_cell;
    //
    // // Получаем из файла решение для ячейки
    // get_flat_deform_and_stress (
    //         ort_slice,
    //         coor_slice,
    //         cell_flat_stress,
    //         cell_flat_deform,
    //         coor_flat_cell,
    //         number_of_approx
    //         );
    //
    // // Строим треангуляцию
    // {
    //     vec<prmt::Point<2>> border;
    //     vec<st> type_border;
    //     give_rectangle_with_border_condition(
    //             border,
    //             type_border,
    //             arr<st, 4>{1,3,2,4},
    //             10,
    //             prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
    //     vec<vec<prmt::Point<2>>> inclusion(1);
    //     cdbl radius = 0.25;
    //     dealii::Point<2> center (0.5, 0.5);
    //     give_circ(inclusion[0], 40, radius, prmt::Point<2>(center));
    //     ::set_grid(domain_cell.grid, border, inclusion, type_border);
    // };
    // dealii::FESystem<3,3> fe 
    //     (dealii::FE_Q<3,3>(1), 3);
    // domain_cell.dof_init (fe);
    //
    // for (auto &&a : cell_stress.content)
    //     for (auto &&b : a)
    //         for (auto &&c : b)
    //             for (auto &&d : c)
    //                 for (auto &&e : d)
    //                     e .resize (domain_cell.dof_handler .n_dofs());
    //
    // for (auto &&a : cell_deform.content)
    //     for (auto &&b : a)
    //         for (auto &&c : b)
    //             for (auto &&d : c)
    //                 for (auto &&e : d)
    //                     e .resize (domain_cell.dof_handler .n_dofs());
    //
    // // Распихивание имеющихся напряжений и деформаций по треангуляции
    // {
    //     auto cell = domain_cell.dof_handler.begin_active();
    //     auto endc = domain_cell.dof_handler.end();
    //     for (; cell != endc; ++cell)
    //     {
    //         for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
    //         {
    //             const dealii::Point<2> p = cell -> vertex(i);
    //             for (st j = 0; j < coor_flat_cell.size(); ++j)
    //             {
    //                 if (
    //                         (std::abs(p(x) - coor_flat_cell[j][x]) < 1.0e-10) and
    //                         (std::abs(p(y) - coor_flat_cell[j][y]) < 1.0e-10)
    //                    )
    //                 {
    //                     v_indx.second = j;
    //                     approx_iteration (number_of_approx, 
    //                             [&cell_stress, size_sol_cell]
    //                             (arr<i32, 3> a, cst nu, cst alpha){
    //                             cell_stress[a][]
    //                     break;
    //                 };
    //             };
    //         };
    //     };
    // };
};

void get_flat_deform_and_stress(
        cst ort_slice,
        cdbl coor_slice,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<vec<dbl>>, 3>, 3>, 3>> &cell_flat_stress,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<vec<dbl>>, 3>, 3>, 3>> &cell_flat_deform,
        vec<vec<dealii::Point<3>>> &coor_flat_cell,
        cst number_of_approx,
        cst n_ref
        )
{
        enum {x, y, z};

        ///////////////////////////////// Техническая часть, добывание из файлов, сформированных решением
        ///////////////////////////////// двух задач выше и формирование линии вдоль которой будут 
        ///////////////////////////////// вычислятся напряжения в дальнейшем.

        st size_sol_cell = 0;
        {
            std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/solution_on_cell_size.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &size_sol_cell, sizeof size_sol_cell);
            in.close ();
        };

        // cst number_of_approx = 3;

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_stress (number_of_approx);
        for (auto &&a : cell_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        OnCell::ArrayWithAccessToVector<arr<arr<vec<dbl>, 3>, 3>> cell_deform (number_of_approx);
        for (auto &&a : cell_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                        e .resize (size_sol_cell);

        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        // std::cout << "cell/stress_"+name+".bin" << std::endl;
                                        std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::ifstream in ("cell/stress_"+name+".bin", std::ios::in | std::ios::binary);
                                        // std::cout << in.is_open() << std::endl;
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_stress[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };

        for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
                                    {
                                        std::ifstream in (
                                                "cell/deform_"+name+".bin", std::ios::in | std::ios::binary);
                                        arr<dbl, 3> tmp;
                                        for (st i = 0; i < size_sol_cell; ++i)
                                        {
                                            dbl tmp = 0.0;
                                            in.read ((char *) &(tmp), sizeof(dbl));
                                            cell_deform[approximation][nu][alpha][i] = tmp;
                                        };
                                        in.close ();
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        // {
        //     // std::cout << "cell/stress_"+name+".bin" << std::endl;
        //     std::ifstream in ("cell/stress_1_0_0_x_x.bin", std::ios::in | std::ios::binary);
        //     // std::cout << in.is_open() << std::endl;
        //     arr<dbl, 3> tmp;
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         // in.read ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
        //         dbl tmp = 0.0;
        //         in.read ((char *) &(tmp), sizeof(dbl));
        //         cell_stress[arr<i32, 3>{1, 0, 0}][x][x][i] = tmp;
        //     };
        //     in.close ();
        // };

        vec<dealii::Point<3>> coor_cell (size_sol_cell);
        {
            std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            // std::ifstream in ("cell/coor_cell.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < size_sol_cell; ++i)
            {
                in.read ((char *) &coor_cell[i](x), sizeof(dbl));
                in.read ((char *) &coor_cell[i](y), sizeof(dbl));
                in.read ((char *) &coor_cell[i](z), sizeof(dbl));
            };
            in.close ();
        };
        // {
        //     FILE *F;
        //     F = fopen("stress_cell_yyy.gpd", "w");
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (i % 3 == y)
        //             fprintf(F,"%f %f %f\n", coor_cell[i](x), coor_cell[i](z), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
        //     };
        //     fclose(F);
        // };
        // for (st i = 0; i < 10; ++i)
        // {
        //     printf("%f %f %f\n", coor_cell[i](x), coor_cell[i](y), coor_cell[i](z));
        // };
        st size_flat_cell = 0;
        // {
        //     // FILE *F;
        //     // F = fopen("stress_cell_flat.gpd", "w");
        //     for (st i = 0; i < size_sol_cell; ++i)
        //     {
        //         if (
        //                 (std::abs(coor_cell[i](ort_slice) - coor_slice) < 1.0e-10)
        //            )
        //
        //         {
        //             if ((i % 3) == y)
        //             {
        //                 // fprintf(F,"%f %f\n", coor_cell[i](x), cell_stress[arr<i32,3>{0,1,0}][y][y][i]);
        //                 ++size_flat_cell;
        //             };
        //         };
        //     };
        //     // fclose(F);
        // };

        // to flat stress
        // for (auto &&a : cell_flat_stress.content)
        //     for (auto &&b : a)
        //         for (auto &&c : b)
        //             for (auto &&d : c)
        //                 for (auto &&e : d)
        //                     for (auto &&j : e)
        //                         j .resize (size_flat_cell);
        // coor_flat_cell .resize(size_flat_cell);
        std::cout << coor_flat_cell.size() << std::endl;
        cst coort_1 = (ort_slice + 1) % 3; 
        cst coort_2 = (ort_slice + 2) % 3; 
        for (st m = 0; m < size_sol_cell; ++m)
        {
            if (
                    (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10)
               )
            {
                auto p = coor_cell[m];
                cst i_x = st(p(coort_1) * ((1 << n_ref)));
                cst i_y = st(p(coort_2) * ((1 << n_ref)));
                // std::cout << i_x << " " << i_y << " " << p(x) * ((1 << n_ref))<< " " <<  p(z) * ((1 << n_ref))
                //     << " " << p(x) << " " << p(z) << std::endl;
                coor_flat_cell[i_x][i_y] = coor_cell[m];
            };
        };

        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10)
                   )
                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                auto p = coor_cell[m];
                                                cst i_x = st(p(coort_1) * ((1 << n_ref)));
                                                cst i_y = st(p(coort_2) * ((1 << n_ref)));
                                                cell_flat_stress[approximation][nu][alpha][beta][i_x][i_y] = 
                                                    cell_stress[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    // if ((m % 3) == z)
                    // {
                    //     // coor_flat_cell[n] = coor_cell[m];
                    //     ++n;
                    // };
                    // if (n == size_flat_cell)
                    //     break;
                };
            };
        };

        // to flat deform
        // for (auto &&a : cell_flat_deform.content)
        //     for (auto &&b : a)
        //         for (auto &&c : b)
        //             for (auto &&d : c)
        //                 for (auto &&e : d)
        //                     for (auto &&j : e)
        //                         j .resize (size_flat_cell);
        // coor_flat_cell .resize(size_flat_cell);
        vec<dbl> move_x(size_flat_cell);
        {
            st n = 0;
            for (st m = 0; m < size_sol_cell; ++m)
            {
                if (
                        (std::abs(coor_cell[m](ort_slice) - coor_slice) < 1.0e-10)
                   )

                {
                    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
                    {
                        for (st i = 0; i < approx_number+1; ++i)
                        {
                            for (st j = 0; j < approx_number+1; ++j)
                            {
                                for (st k = 0; k < approx_number+1; ++k)
                                {
                                    if ((i+j+k) == approx_number)
                                    {
                                        arr<i32, 3> approximation = {i, j, k};
                                        for (st nu = 0; nu < 3; ++nu)
                                        {
                                            for (st alpha = 0; alpha < 3; ++alpha)
                                            {
                                                st beta = m % 3;
                                                // if ((i == 1) and (j == 0) and (k == 0) and (nu == x) and (alpha == x))
                                                // for (st beta = 0; beta < 3; ++beta)
                                                // {
                                                //     if ((m % 3) == beta)
                                                //     {
                                                auto p = coor_cell[m];
                                                cst i_x = st(p(coort_1) * ((1 << n_ref)));
                                                cst i_y = st(p(coort_2) * ((1 << n_ref)));
                                                cell_flat_deform[approximation][nu][alpha][beta][i_x][i_y] = 
                                                    cell_deform[approximation][nu][alpha][m];
                                                        // printf("%d %d %d %d %d\n", i, j, k, nu, alpha);
                                                //     };
                                                // };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
        //                             // printf("%d\n", n);
                    // if ((m % 3) == z)
                    // {
                    //     // coor_flat_cell[n] = coor_cell[m];
                    //     ++n;
                    // };
                    // if (n == size_flat_cell)
                    //     break;
                };
            };
        };
        {
            std::ofstream f("stress_xxxx.gpd", std::ios::out);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
            for (st j = 0; j < (1 << n_ref) + 1; ++j)
            {
                f << coor_flat_cell[i][j](x) << " " << coor_flat_cell[i][j](z) << " " << 
                    cell_flat_stress[arr<i32,3>{1,0,0}][x][x][x][i][j] << std::endl;
                // f << "dfdgdf" << std::endl;
            };
            f.close ();
        };

        // printf("size_flat %d %d\n", size_flat_hole, size_flat_cell);
        // printf("size %d %d\n", size_sol_hole, size_sol_cell);
    
};

template <cst n_ref>
void solve_ring_problem_3d (cst flag, cdbl H, cdbl W, cdbl Ri, cst n_rad_cell, cdbl P)
{
    if (flag)
    {
        enum {x, y, z};

        cdbl Ro = Ri + W;

        Domain<3> domain;
        vec<dealii::Point<2>> radius_vector;
        vec<dealii::Point<2>> radius_vector_cell;
        {
            {
                auto center = dealii::Point<2>(0.0, 0.0);
                // dealii::GridGenerator::cylinder_shell(domain.grid, 1.0, Ri, Ro, 1 << 4, 1);
                // dealii::GridGenerator::cylinder_shell(domain.grid, length, 1.0, 2.0);
                dealii::GridGenerator::cylinder_shell(domain.grid, H, Ri, Ro, n_rad_cell, 1.0);
                auto cell = domain.grid .begin_active();
                auto end_cell = domain.grid .end();
                for (; cell != end_cell; ++cell)
                {
                    // cell->set_refinement_case (
                    //         dealii::RefinementCase<3>(dealii::RefinementPossibilities<3>::Possibilities::cut_x));
                    // dealii::RefinementCase<3> ref_case(2);
                    // cell->set_refine_flag (ref_case);
                    for (st i = 0; i < 6; ++i)
                    {
                        if (cell->face(i)->at_boundary())
                        {
                            auto p = cell->face(i)->center();
                            auto p2d = dealii::Point<2>(p(0), p(1));
                            if (std::abs(center.distance(p2d)) - Ri < 1.0e-8)
                            {
                                radius_vector .push_back(p2d);
                                cell->face(i)->set_boundary_indicator(radius_vector.size()+3);
                            };
                            if (std::abs(p(2) - 0.0) < 1.0e-5)
                            {
                                cell->face(i)->set_boundary_indicator(1);
                            };
                            if (std::abs(p(2) - H) < 1.0e-5)
                            {
                                cell->face(i)->set_boundary_indicator(2);
                            };
                        };
                        // auto p = cell->face(i)->center();
                        // auto p2d = dealii::Point<2>(p(0), p(1));
                        // if (std::abs(p(2) - length / 2.0) < 1.0e-8)
                        // {
                        //     // cell->face(i)->set_boundary_indicator(3);
                        // };
                    };

                    dealii::Point<2> midle_p(0.0, 0.0);
                    for (size_t i = 0; i < 8; ++i)
                    {
                        midle_p(0) += cell->vertex(i)(0);
                        midle_p(1) += cell->vertex(i)(1);
                    };
                    midle_p(0) /= 8.0;
                    midle_p(1) /= 8.0;
                    cell->set_material_id(radius_vector_cell.size());
                    radius_vector_cell .push_back(midle_p);
                };
                // domain.grid.refine_global(n_ref);
                // domain.grid.execute_coarsening_and_refinement();
            };
            {
                for (st i = 0; i < n_ref; ++i)
                {

                    auto cell = domain.grid .begin_active();
                    auto end_cell = domain.grid .end();
                    for (; cell != end_cell; ++cell)
                    {
                        dealii::RefinementCase<3> ref_case(6);
                        cell->set_refine_flag (ref_case);
                    };
                    domain.grid.execute_coarsening_and_refinement();
                };
            };
        };

        dealii::FESystem<3,3> fe(dealii::FE_Q<3,3>(1), 3);
        domain.dof_init (fe);

        SystemsLinearAlgebraicEquations slae;
        ATools ::trivial_prepare_system_equations (slae, domain);

        LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());

        ATools::FourthOrderTensor C;
        {
            ATools::FourthOrderTensor Cxy;
            std::ifstream in ("meta_coef.bin", std::ios::in | std::ios::binary);
            in.read ((char *) &Cxy, sizeof Cxy);
            in.close ();
            arr<arr<dbl,3>,3> A = {
                arr<dbl,3>{1.0, 0.0, 0.0},
                arr<dbl,3>{0.0, 0.0, -1.0},
                arr<dbl,3>{0.0, 1.0, 0.0}
            };
            for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
            for (st k = 0; k < 3; ++k)
            for (st l = 0; l < 3; ++l)
            {
                C[i][j][k][l] = 0.0;

                for (st a = 0; a < 3; ++a)
                for (st b = 0; b < 3; ++b)
                for (st c = 0; c < 3; ++c)
                for (st d = 0; d < 3; ++d)
                {
                    C[i][j][k][l] += 
                        A[i][a]*A[j][b]*A[k][c]*A[l][d]*Cxy[a][b][c][d];

                };
            };
        };

        element_matrix.C .resize (radius_vector_cell.size());
        for (st n = 0; n < radius_vector_cell.size(); ++n)
        {
            cdbl hypo = sqrt(
                    radius_vector_cell[n](x)*radius_vector_cell[n](x)+
                    radius_vector_cell[n](y)*radius_vector_cell[n](y)
                    );
            cdbl sin = radius_vector_cell[n](y)/hypo;
            cdbl cos = radius_vector_cell[n](x)/hypo;
            arr<arr<dbl,3>,3> A = {
                arr<dbl,3>{cos, -sin, 0.0},
                arr<dbl,3>{sin,  cos, 0.0},
                arr<dbl,3>{0.0,  0.0, 1.0}
            };

            for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
            for (st k = 0; k < 3; ++k)
            for (st l = 0; l < 3; ++l)
            {
                element_matrix.C[n][i][j][k][l] = 0.0;

                for (st a = 0; a < 3; ++a)
                for (st b = 0; b < 3; ++b)
                for (st c = 0; c < 3; ++c)
                for (st d = 0; d < 3; ++d)
                {
                    element_matrix.C[n][i][j][k][l] += 
                        // C[i][j][k][l];
                        A[i][a]*A[j][b]*A[k][c]*A[l][d]*C[a][b][c][d];

                };
            };

            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[n]);
        };
            // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(element_matrix.C[0]);
            // exit(1);

        arr<std::function<dbl (const dealii::Point<3>&)>, 3> func {
            [] (const dealii::Point<3>) {return 0.0;},
            [] (const dealii::Point<3>) {return 0.0;},
            [] (const dealii::Point<3>) {return 0.0;}
        };
        SourceVector<3> element_rhsv (func, domain.dof_handler.get_fe());

        Assembler ::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler);

        // cdbl P = 1.0;
        vec<BoundaryValueVector<3>> bound (radius_vector.size()+3);
        bound[0].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[0].boundary_id   = 1;
        bound[0].boundary_type = TBV::Neumann;
        bound[1].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[1].boundary_id   = 2;
        bound[1].boundary_type = TBV::Neumann;
        bound[2].function      = [] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
        bound[2].boundary_id   = 3;
        // bound[2].boundary_type = TBV::Neumann;
        bound[2].boundary_type = TBV::Dirichlet;

        for (st i = 0; i < radius_vector.size(); ++i)
        {
            cdbl a = radius_vector[i](0);
            cdbl b = radius_vector[i](1);
            cdbl tmp = 1/std::sqrt(a*a + b*b);
            cdbl Px = P*a*tmp;
            cdbl Py = P*b*tmp;
            // printf("%f %f\n", Px, Py);
            bound[i+3].function      = [Px, Py] (const dealii::Point<3> &p) {return arr<dbl, 3>{Px, Py, 0.0};};
            // bound[i+3].function      = [Px, Py] (const dealii::Point<3> &p) {return arr<dbl, 3>{0.0, 0.0, 0.0};};
            // bound[i+3].function      = [P] (const dealii::Point<3> &p) {
            //     cdbl a = p(0);
            //     cdbl b = p(1);
            //     cdbl tmp = 1/std::sqrt(a*a + b*b);
            //     cdbl Px = P*a*tmp;
            //     cdbl Py = P*b*tmp;
            //     return arr<dbl, 3>{Px, Py, 0.0};
            // };
            bound[i+3].boundary_id   = i+4;
            bound[i+3].boundary_type = TBV::Neumann;
            // bound[i].boundary_type = TBV::Dirichlet;
        };

        for (auto b : bound)
            ATools ::apply_boundary_value_vector<3> (b) .to_slae (slae, domain);

        // Зажимаем точки в середине трубы по оси z
        // if (0)
        {
            std::map<u32, dbl> list_boundary_values;
            auto cell = domain.dof_handler .begin_active();
            auto end_cell = domain.dof_handler .end();
            for (; cell != end_cell; ++cell)
            {
                for (st i = 0; i < 8; ++i)
                {
                    if (std::abs(cell->vertex(i)(z) - H / 2.0 ) < 1.0e-5)
                    {
                        cst v = cell->vertex_dof_index(i, z);
                        if (list_boundary_values.find(v) == list_boundary_values.end())
                        {
                            std::pair<u32, dbl> bv (v, 0.0);
                            list_boundary_values.insert(bv);
                            // std::pair<u32, dbl> bv1 (cell->vertex_dof_index(i, x), 0.0);
                            // std::pair<u32, dbl> bv2 (cell->vertex_dof_index(i, y), 0.0);
                            // std::pair<u32, dbl> bv3 (cell->vertex_dof_index(i, z), 0.0);
                            // list_boundary_values.insert(bv1);
                            // list_boundary_values.insert(bv2);
                            // list_boundary_values.insert(bv3);
                        }; 
                    };
                    if (
                            // (std::abs(std::abs(cell->vertex(i)(x)) - Ro) < 1.0e-5) and
                            (std::abs(cell->vertex(i)(y)) < 1.0e-5))
                    {
                        cst v = cell->vertex_dof_index(i, y);
                        if (list_boundary_values.find(v) == list_boundary_values.end())
                        {
                            list_boundary_values.insert(std::pair<u32, dbl>(v, 0.0));
                            // list_boundary_values.insert(std::pair<u32, dbl>(cell->vertex_dof_index(i, y), 0.0));
                        };
                    };
                    if (
                            // (std::abs(std::abs(cell->vertex(i)(y)) - Ro) < 1.0e-5) and
                            (std::abs(cell->vertex(i)(x)) < 1.0e-5))
                    {
                        cst v = cell->vertex_dof_index(i, x);
                        if (list_boundary_values.find(v) == list_boundary_values.end())
                        {
                            list_boundary_values.insert(std::pair<u32, dbl>(v, 0.0));
                            // list_boundary_values.insert(std::pair<u32, dbl>(cell->vertex_dof_index(i, y), 0.0));
                        };
                    };
                };
            };
            dealii::MatrixTools::apply_boundary_values (
                    list_boundary_values,
                    slae.matrix,
                    slae.solution,
                    slae.rhsv);
        };

        puts("111111111");
        dealii::SolverControl solver_control (1000000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        // dealii::SolverRelaxation<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                // ,dealii::RelaxationBlockJacobi<dealii::SparseMatrix<dbl>>::AdditionalData(1.0)
                );
        

        // EPTools ::print_move<3> (slae.solution, domain.dof_handler, "move.gpd");
        EPTools ::print_move<3> (slae.solution, domain.dof_handler, "move.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_temperature<3> (slae.solution, domain.dof_handler, "move.vtk", dealii::DataOutBase::vtk);
        // HCPTools ::print_temperature_slice (slae.solution, domain.dof_handler, "temperature_slice.gpd", x, len_rod);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice_z.gpd", z, 0.0);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice_z_l.gpd", z, H / 2.0);
        EPTools ::print_move_slice (slae.solution, domain.dof_handler, "move_slice_y.gpd", y, 0.0);
        dealii::Vector<dbl> anal(slae.solution.size());
        dealii::Vector<dbl> anal_deform(slae.solution.size());
        {
            cdbl C1 = 0.75 * P * Ri*Ri / (Ro*Ro-Ri*Ri);
            cdbl C2 = 1.25 * P * Ri*Ri * Ro*Ro / (Ro*Ro-Ri*Ri);
            cdbl C3 = -0.5 * P * Ri*Ri / (Ro*Ro-Ri*Ri);
            auto cell = domain.dof_handler .begin_active();
            auto end_cell = domain.dof_handler .end();
            for (; cell != end_cell; ++cell)
            {
                for (st i = 0; i < 8; ++i)
                {
                    auto p = cell->vertex(i);
                    cdbl r = std::pow(p(x)*p(x)+p(y)*p(y), 0.5);
                    anal(cell->vertex_dof_index(i,x)) = C1 * r + C2 / r;
                    anal(cell->vertex_dof_index(i,y)) = 0.0;
                    anal(cell->vertex_dof_index(i,z)) = C3 * p(z);
                    anal_deform(cell->vertex_dof_index(i,x)) = C1 + C2 / (r*r);
                    anal_deform(cell->vertex_dof_index(i,y)) = C1 + C2 / (r*r);
                    anal_deform(cell->vertex_dof_index(i,z)) = C3;
                    // anal(cell->vertex_dof_index(i,x)) = C1 + C2 / (r*r);
                    // anal(cell->vertex_dof_index(i,y)) = C1 + C2 / (r*r);
                    // anal(cell->vertex_dof_index(i,z)) = C3;
                };
            };
            std::cout << C1 << " " << C2 << " " << C3 << std::endl;
        };
        EPTools ::print_move<3> (anal, domain.dof_handler, "anal_move.vtk", dealii::DataOutBase::vtk);
        EPTools ::print_move_slice (anal, domain.dof_handler, "anal_move_slice_z.gpd", z, 0.0);
        EPTools ::print_move_slice (anal, domain.dof_handler, "anal_move_slice_y.gpd", y, 0.0);
        EPTools ::print_move_slice (anal_deform, domain.dof_handler, "anal_deform_slice_z.gpd", z, 0.0);
        EPTools ::print_move_slice (anal_deform, domain.dof_handler, "anal_deform_slice_y.gpd", y, 0.0);

        puts("!!!!!!!!!!1");
        arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v;
        arr<arr<dealii::Point<2>,(1 << n_ref) + 1>,(1 << n_ref) + 1> coor;
        {
            auto cell = domain.dof_handler .begin_active();
            auto end_cell = domain.dof_handler .end();
            for (; cell != end_cell; ++cell)
            {
                for (st i = 0; i < 8; ++i)
                {
                    auto p = cell->vertex(i);
                    if ((p(x) > 0.0) and (std::abs(p(y)) < 1.0e-5))
                    {
                        p(x) -= Ri;
                        p(x) /= W;
                        p(z) /= H;
                        cst i_x = st(p(x) * ((1 << n_ref)));
                        cst i_y = st(p(z) * ((1 << n_ref)));
                        // std::cout << i_x << " " << i_y << std::endl;
                        v[i_x][i_y][x] = slae.solution((cell->vertex_dof_index (i, x)));
                        v[i_x][i_y][y] = slae.solution((cell->vertex_dof_index (i, y)));
                        v[i_x][i_y][z] = slae.solution((cell->vertex_dof_index (i, z)));
                        coor[i_x][i_y](x) = cell->vertex(i)(x);
                        coor[i_x][i_y](y) = cell->vertex(i)(z);
                    };
                };
            };
        };
        puts("!!!!!!!!!!1");

        {
            std::ofstream f("move_ring.gpd", std::ios::out);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
            {
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    f << coor[i][j](x) << " " << coor[i][j](y) << " " << v[i][j][x] << std::endl;
                };
            };
            f.close ();
        };

        {
            std::ofstream out ("ring_coor.bin", std::ios::out | std::ios::binary);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    out.write ((char *) &coor[i][j](x), sizeof(dbl));
                    out.write ((char *) &coor[i][j](y), sizeof(dbl));
                };
            out.close ();
        };

        {
            std::ofstream out ("ring_move.bin", std::ios::out | std::ios::binary);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    out.write ((char *) &v[i][j][x], sizeof(dbl));
                    out.write ((char *) &v[i][j][y], sizeof(dbl));
                    out.write ((char *) &v[i][j][z], sizeof(dbl));
                };
            out.close ();
        };
        puts("!!!!!!!!!!1");

        arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v_anal;
        {
            auto cell = domain.dof_handler .begin_active();
            auto end_cell = domain.dof_handler .end();
            for (; cell != end_cell; ++cell)
            {
                for (st i = 0; i < 8; ++i)
                {
                    auto p = cell->vertex(i);
                    if ((p(x) > 0.0) and (std::abs(p(y)) < 1.0e-5))
                    {
                        p(x) -= Ri;
                        p(x) /= W;
                        p(z) /= H;
                        cst i_x = st(p(x) * ((1 << n_ref)));
                        cst i_y = st(p(z) * ((1 << n_ref)));
                        v_anal[i_x][i_y][x] = anal((cell->vertex_dof_index (i, x)));
                        v_anal[i_x][i_y][y] = anal((cell->vertex_dof_index (i, y)));
                        v_anal[i_x][i_y][z] = anal((cell->vertex_dof_index (i, z)));
                    };
                };
            };
        };
        puts("!!!!!!!!!!1");
        {
            std::ofstream out ("ring_move_anal.bin", std::ios::out | std::ios::binary);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    out.write ((char *) &v_anal[i][j][x], sizeof(dbl));
                    out.write ((char *) &v_anal[i][j][y], sizeof(dbl));
                    out.write ((char *) &v_anal[i][j][z], sizeof(dbl));
                };
            out.close ();
        };
    };
};

template <cst n_ref, cst n_cell_ref>
void calculate_real_stress_in_ring(
        cst flag, cst ratio, cdbl width, cdbl Ri, cdbl Ro,
        cst Ncx, cst Npx, cst Npy)
{
    if (flag)
    {  
        enum {x, y, z};

        cst Ncy = Ncx * ratio;
        cdbl bx = width / Ncx;
        cdbl by = 1.0 / Ncy;
        cdbl dx = width / (Npx-1);
        cdbl dy = 1.0 / (Npy-1);

        arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v;
        arr<arr<dealii::Point<2>,(1 << n_ref) + 1>,(1 << n_ref) + 1> coor;

        {
            std::ifstream in ("ring_coor.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    in.read ((char *) &coor[i][j](x), sizeof(dbl));
                    in.read ((char *) &coor[i][j](y), sizeof(dbl));
                };
            in.close ();
        };

        {
            std::ifstream in ("ring_move.bin", std::ios::in | std::ios::binary);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    in.read ((char *) &v[i][j][x], sizeof(dbl));
                    in.read ((char *) &v[i][j][y], sizeof(dbl));
                    in.read ((char *) &v[i][j][z], sizeof(dbl));
                };
            in.close ();
        };
        cdbl Vxx = (v[(1 << n_ref)][0][x] - v[0][0][x]) / width;
        cdbl Vzz = (v[0][(1 << n_ref)][z] - v[0][0][z]) / 1.0;
        std::cout << Vxx << " " << Vzz << std::endl;
        std::cout << v[(1 << n_ref)][0][x]<< " " << v[0][0][x] << std::endl;

        {
            std::ofstream f("move_ring_1.gpd", std::ios::out);
            for (st i = 0; i < (1 << n_ref) + 1; ++i)
            {
                for (st j = 0; j < (1 << n_ref) + 1; ++j)
                {
                    f << coor[i][j](x) << " " << coor[i][j](y)
                        << " " << v[i][j][x]
                        << " " << v[i][j][y]
                        << " " << v[i][j][z]  << std::endl;
                };
            };
            f.close ();
        };
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<vec<dbl>>, 3>, 3>, 3>> cell_flat_stress(3);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<vec<dbl>>, 3>, 3>, 3>> cell_flat_deform(3);
        vec<vec<dealii::Point<3>>> coor_flat_cell;
        //
        for (auto &&a : cell_flat_deform.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                            {
                                j .resize ((1 << n_cell_ref) + 1);
                                for (auto &&o :j)
                                    o .resize ((1 << n_cell_ref) + 1);
                            };

        for (auto &&a : cell_flat_stress.content)
            for (auto &&b : a)
                for (auto &&c : b)
                    for (auto &&d : c)
                        for (auto &&e : d)
                            for (auto &&j : e)
                            {
                                j .resize ((1 << n_cell_ref) + 1);
                                for (auto &&o :j)
                                    o .resize ((1 << n_cell_ref) + 1);
                            };

        coor_flat_cell .resize((1 << n_cell_ref) + 1);
        for (auto &&o : coor_flat_cell)
            o .resize ((1 << n_cell_ref) + 1);


        get_flat_deform_and_stress (
                y,
                0.5,
                cell_flat_stress,
                cell_flat_deform,
                coor_flat_cell,
                2,
                n_cell_ref
                );

        auto stress_in_cell = [&cell_flat_stress, &coor_flat_cell] 
            (cst i, cst j, arr<i32, 3> k, cst nu, cst alpha, cst beta, cdbl px, cdbl py){
            arr<prmt::Point<2>, 4> points = {
                prmt::Point<2>(coor_flat_cell[i][j](0),     coor_flat_cell[i][j](2)),
                prmt::Point<2>(coor_flat_cell[i+1][j](0),   coor_flat_cell[i+1][j](2)),
                prmt::Point<2>(coor_flat_cell[i+1][j+1](0), coor_flat_cell[i+1][j+1](2)),
                prmt::Point<2>(coor_flat_cell[i][j+1](0),   coor_flat_cell[i][j+1](2))};
            // std::cout << i << " " << j << " ";
            // for (st i = 0; i < 4; ++i)
            // {
            //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
            // };
            // std::cout << std::endl;
            arr<dbl, 4> values = {
                cell_flat_stress[k][nu][alpha][beta][i][j],
                cell_flat_stress[k][nu][alpha][beta][i+1][j],
                cell_flat_stress[k][nu][alpha][beta][i+1][j+1],
                cell_flat_stress[k][nu][alpha][beta][i][j+1]};

                Scalar4PointsFunc<2> func(points, values);

                // return   cell_flat_stress[k][nu][alpha][beta][i][j];
                return func(prmt::Point<2>(px, py));
            };

        auto move_in_macro = [&v, &coor] 
            (cst i, cst j, cst alpha, cst beta, cdbl px, cdbl py){
            arr<prmt::Point<2>, 4> points = {
                prmt::Point<2>(coor[i][j](0),     coor[i][j](1)),
                prmt::Point<2>(coor[i+1][j](0),   coor[i+1][j](1)),
                prmt::Point<2>(coor[i+1][j+1](0), coor[i+1][j+1](1)),
                prmt::Point<2>(coor[i][j+1](0),   coor[i][j+1](1))};
            // std::cout << i << " " << j << " ";
            // for (st i = 0; i < 4; ++i)
            // {
            //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
            // };
            // std::cout << std::endl;
            arr<dbl, 4> values = {
                v[i][j][alpha],
                v[i+1][j][alpha],
                v[i+1][j+1][alpha],
                v[i][j+1][alpha]};

                Scalar4PointsFunc<2> func(points, values);

                dbl res = 0.0; 
                switch (beta) {
                    case 0: 
                        res = func.dx(px, py);
                        break;
                    case 1:
                        res = func(px, py);// / std::pow(px*px + py*py, 0.5);
                        break;
                    case 2:
                        res = func.dy(px, py);
                        break;
                };
                return res; //func;//.dy(prmt::Point<2>(px, py));
            };

        arr<arr<vec<vec<dbl>>,3>,3> grad;
        for (st i = 0; i < 3; ++i)
        for (st j = 0; j < 3; ++j)
        {
            grad[i][j] .resize (Npx); 
            for (st k = 0; k < Npx; ++k)
            {
                grad[i][j][k] .resize (Npy); 
            };
        };
        for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            for (st ort_2 = 0; ort_2 < 3; ++ort_2)
        {
                    std::cout << ort_1 << " " << ort_2 << std::endl;
            for (st i = 0; i < Npx-1; ++i)
            {
                for (st j = 0; j < Npy-1; ++j)
                {
                    cdbl coor_x = Ri + dx * i;
                    cdbl coor_z = dy * j;

                    cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                    cst i_z = st(coor_z * (1 << n_ref));

                    dbl ksi_x = coor_x - Ri;
                    while (ksi_x > bx)
                        ksi_x -= bx;
                    ksi_x /= bx;
                    cst i_ksi_x = st(ksi_x * (1 << n_cell_ref));

                    dbl ksi_z = coor_z;
                    while (ksi_z > by)
                        ksi_z -= by;
                    ksi_z /= by;
                    cst i_ksi_z = st(ksi_z * (1 << n_cell_ref));
                    // std::cout <<
                    //     "x=" << coor_x << " y=" << coor_z <<
                    //     // " px=" << coor_flat_cell[i][j](x) << " py=" <<  coor_flat_cell[i][j](z) <<
                    //     " i_x=" << i_x << " i_z= " << i_z << 
                    //     " i_ksi_x=" << i_ksi_x << " i_ksi_z=" << i_ksi_z << 
                    //     std::endl;

                    // grad[0][0][i][j] = stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1,0,0}, x, x, x, ksi_x, ksi_z);
                    // grad[0][0][i][j] = stress_in_cell (i, j, arr<i32, 3>{1,0,0}, x, x, x, coor_x, coor_z);
                    grad[ort_1][ort_2][i][j] =
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, x, x, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, x, y, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, x, z, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 1, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, y, x, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 1, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, y, y, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 1, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, y, z, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, z, x, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, z, y, coor_x, coor_z) +
                        stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        move_in_macro(i_x, i_z, z, z, coor_x, coor_z);

                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        // move_in_macro(i_x, i_z, x).dy(coor_x, coor_z) +
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // move_in_macro(i_x, i_z, z).dx(coor_x, coor_z) +
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // move_in_macro(i_x, i_z, z).dy(coor_x, coor_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                    //     move_in_macro(i_z, i_x, x).dx(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                    //     move_in_macro(i_z, i_x, x).dy(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                    //     move_in_macro(i_z, i_x, z).dx(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                    //     move_in_macro(i_z, i_x, z).dy(coor_x, coor_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);// *
                        // Vxx;// +
                        // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // Vzz;
                    // grad[ort_1][ort_2][i][j] =1.0;
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, x, x, ksi_x, ksi_z);
                        // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);
                        // move_in_macro(i_x, i_z, ort_1).dx(coor_x, coor_z);
                        // move_in_macro(i_x, i_z, ort_1)(coor_x, coor_z);
                };
            };
        };
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < 2; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                for (st beta = 0; beta < 3; ++beta)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+
                                        str("_")+ort[nu]+str("_")+ort[alpha]+str("_")+ort[beta];
                                    {
                                        std::ofstream out ("ring/stress_cell_"+name+".gpd", std::ios::out);
            for (st m = 0; m < Npx-1; ++m)
            {
                for (st n = 0; n < Npy-1; ++n)
                {
                    cdbl coor_x = Ri + dx * n;
                    cdbl coor_z = dy * m;

                    cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                    cst i_z = st(coor_z * (1 << n_ref));

                    dbl ksi_x = coor_x - Ri;
                    while (ksi_x > bx)
                        ksi_x -= bx;
                    ksi_x /= bx;
                    cst i_ksi_x = st(ksi_x * (1 << n_cell_ref));

                    dbl ksi_z = coor_z;
                    while (ksi_z > by)
                        ksi_z -= by;
                    ksi_z /= by;
                    cst i_ksi_z = st(ksi_z * (1 << n_cell_ref));

                    out << coor_x << " " << coor_z << " " << 
                        stress_in_cell (i_ksi_z, i_ksi_x, approximation, nu, alpha, beta, ksi_x, ksi_z)
                        << std::endl;
                };
            };
                                        out.close ();
                                    };
                                };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        {
            arr<str, 3> ort = {"x", "y", "z"};
            for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            {
                for (st ort_2 = 0; ort_2 < 3; ++ort_2)
                {
                    {
                        std::ofstream f(str("ring/stress_macro_") + ort[ort_1] + ort[ort_2] +".gpd", std::ios::out);
                        for (st i = 0; i < Npx-1; ++i)
                        {
                            for (st j = 0; j < Npy-1; ++j)
                            {
                                cdbl coor_x = Ri + dx * i;
                                cdbl coor_z = dy * j;

                                cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                                cst i_z = st(coor_z * (1 << n_ref));
                                f << coor_x << " " << coor_z << " " << 
                                    move_in_macro(i_x, i_z, ort_1, ort_2, coor_x, coor_z)
                                    << std::endl;
                            };
                        };
                        f.close ();
                    };
                };
            };
        };
        {
            arr<str, 3> ort = {"x", "y", "z"};
            for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            {
                for (st ort_2 = 0; ort_2 < 3; ++ort_2)
                {
                    std::ofstream f(str("ring/stress_real_") + ort[ort_1] + ort[ort_2] + ".gpd", std::ios::out);
                    for (st n = 0; n < Npx-1; ++n)
                    {
                        for (st m = 0; m < Npy-1; ++m)
                        {
                            // f << coor_flat_cell[i][j](x) << " " << coor_flat_cell[i][j](z) << " " << grad[0][0][i][j] << std::endl;
                            cdbl coor_x = Ri + dx * n;
                            cdbl coor_z = dy * m;

                            cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                            cst i_z = st(coor_z * (1 << n_ref));

                            dbl ksi_x = coor_x - Ri;
                            while (ksi_x > bx)
                                ksi_x -= bx;
                            ksi_x /= bx;
                            cst i_ksi_x = st(ksi_x * (1 << n_cell_ref));

                            dbl ksi_z = coor_z;
                            while (ksi_z > by)
                                ksi_z -= by;
                            ksi_z /= by;
                            cst i_ksi_z = st(ksi_z * (1 << n_cell_ref));

                            f << coor_x << " " << coor_z << " " << 
                                grad[ort_1][ort_2][n][m]
                                // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, 1, 1, ksi_x, ksi_z)
                                // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z)
                                << std::endl;
                            // cell_flat_stress[arr<i32,3>{1,0,0}][x][x][x][i][j] << std::endl;
                        };
                    };
                    f.close ();
                };
            };
        };
    };
};

template <cst n_ref>
void calculate_real_stress_in_ring_arbitrary_grid(
        cst flag, cdbl ratio, cdbl Ri, cdbl Ro,
        cst Ncx, cst Npx, cst Npy)
{
    if (flag)
    {  
        enum {x, y, z};

        cdbl width = 1.0 / ratio;
        cst Ncy = Ncx * ratio;
        cdbl bx = width / Ncx;
        cdbl by = 1.0 / Ncy;
        cdbl dx = width / (Npx-1);
        cdbl dy = 1.0 / (Npy-1);

        arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v;
        arr<arr<dealii::Point<2>,(1 << n_ref) + 1>,(1 << n_ref) + 1> coor;

        // Получаем из файла решение в кольце
        {
            {
                std::ifstream in ("ring_coor.bin", std::ios::in | std::ios::binary);
                for (st i = 0; i < (1 << n_ref) + 1; ++i)
                    for (st j = 0; j < (1 << n_ref) + 1; ++j)
                    {
                        in.read ((char *) &coor[i][j](x), sizeof(dbl));
                        in.read ((char *) &coor[i][j](y), sizeof(dbl));
                    };
                in.close ();
            };

            {
                std::ifstream in ("ring_move.bin", std::ios::in | std::ios::binary);
                for (st i = 0; i < (1 << n_ref) + 1; ++i)
                    for (st j = 0; j < (1 << n_ref) + 1; ++j)
                    {
                        in.read ((char *) &v[i][j][x], sizeof(dbl));
                        in.read ((char *) &v[i][j][y], sizeof(dbl));
                        in.read ((char *) &v[i][j][z], sizeof(dbl));
                    };
                in.close ();
            };
        };

        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_stress(3);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_deform(3);
        vec<dealii::Point<3>> coor_flat_cell;

        // Получаем из файла решение для ячейки
        get_flat_deform_and_stress (
                z,
                0.5,
                cell_flat_stress,
                cell_flat_deform,
                coor_flat_cell,
                2
                );

        dealii::Triangulation<2> grid;

        // Строим треангуляцию
        {
            vec<prmt::Point<2>> border;
            vec<st> type_border;
            give_rectangle_with_border_condition(
                    border,
                    type_border,
                    arr<st, 4>{1,3,2,4},
                    10,
                    prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
            vec<vec<prmt::Point<2>>> inclusion(1);
            cdbl radius = 0.25;
            dealii::Point<2> center (0.5, 0.5);
            give_circ(inclusion[0], 40, radius, prmt::Point<2>(center));
            ::set_grid(grid, border, inclusion, type_border);
        };

        std::map<st, st> v_in_cell;

        // Получаем соответствие между точками триангуляции и точками решения
        {
            auto cell = grid.begin_active();
            auto endc = grid.end();
            for (; cell != endc; ++cell)
            {
                for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
                {
                    cst v = cell -> vertex_index(i);
                    const dealii::Point<2> p = cell -> vertex(i);
                    if (v_in_cell.find(v) == v_in_cell.end())
                    {
                        std::pair<st, st> v_indx;
                        v_indx.first = v;
                        for (st j = 0; j < coor_flat_cell.size(); ++j)
                        {
                            if (
                                    (std::abs(p(x) - coor_flat_cell[j][x]) < 1.0e-10) and
                                    (std::abs(p(y) - coor_flat_cell[j][y]) < 1.0e-10)
                               )
                            {
                                v_indx.second = j;
                                break;
                            };
                        };
                        v_in_cell .insert (v_indx);
                    };
                };
            };
        };


        auto stress_in_cell = [&cell_flat_stress, &coor_flat_cell, &grid, &v_in_cell] 
            (arr<i32, 3> k, cst nu, cst alpha, cst beta, cdbl px, cdbl py){
                arr<st, 4> v;
                {
                    auto cell = grid.begin_active();
                    auto endc = grid.end();
                    for (; cell != endc; ++cell)
                    {
                        if (point_in_cell(dealii::Point<2>(px, py), cell))
                        {
                            v[0] = v_in_cell[cell->vertex_index(0)];
                            v[1] = v_in_cell[cell->vertex_index(1)];
                            v[2] = v_in_cell[cell->vertex_index(3)];
                            v[3] = v_in_cell[cell->vertex_index(2)];
                            break;
                        };
                    };
                };
                arr<prmt::Point<2>, 4> points = {
                    prmt::Point<2>(coor_flat_cell[v[0]](0), coor_flat_cell[v[0]](1)),
                    prmt::Point<2>(coor_flat_cell[v[1]](0), coor_flat_cell[v[1]](1)),
                    prmt::Point<2>(coor_flat_cell[v[2]](0), coor_flat_cell[v[2]](1)),
                    prmt::Point<2>(coor_flat_cell[v[3]](0), coor_flat_cell[v[3]](1))
                };
                // std::cout << i << " " << j << " ";
                // for (st i = 0; i < 4; ++i)
                // {
                //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
                // };
                // std::cout << std::endl;
                arr<dbl, 4> values = {
                    cell_flat_stress[k][nu][alpha][beta][v[0]],
                    cell_flat_stress[k][nu][alpha][beta][v[1]],
                    cell_flat_stress[k][nu][alpha][beta][v[2]],
                    cell_flat_stress[k][nu][alpha][beta][v[3]]};

                Scalar4PointsFunc<2> func(points, values);

                // return   cell_flat_stress[k][nu][alpha][beta][i][j];
                return func(prmt::Point<2>(px, py));
            };

        auto deform_in_macro = [&v, &coor] 
            (cst i, cst j, cst alpha, cst beta, cdbl px, cdbl py){
            arr<prmt::Point<2>, 4> points = {
                prmt::Point<2>(coor[i][j](0),     coor[i][j](1)),
                prmt::Point<2>(coor[i+1][j](0),   coor[i+1][j](1)),
                prmt::Point<2>(coor[i+1][j+1](0), coor[i+1][j+1](1)),
                prmt::Point<2>(coor[i][j+1](0),   coor[i][j+1](1))};
            // std::cout << i << " " << j << " ";
            // for (st i = 0; i < 4; ++i)
            // {
            //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
            // };
            // std::cout << std::endl;
            arr<dbl, 4> values = {
                v[i][j][alpha],
                v[i+1][j][alpha],
                v[i+1][j+1][alpha],
                v[i][j+1][alpha]};

                Scalar4PointsFunc<2> func(points, values);

                dbl res = 0.0; 
                switch (beta) {
                    case 0: 
                        res = func.dx(px, py);
                        break;
                    case 1:
                        res = func(px, py) / std::pow(px*px + py*py, 0.5);
                        break;
                    case 2:
                        res = func.dy(px, py);
                        break;
                };
                return res; //func;//.dy(prmt::Point<2>(px, py));
            };

        arr<arr<vec<vec<dbl>>,3>,3> grad;
        for (st i = 0; i < 3; ++i)
        for (st j = 0; j < 3; ++j)
        {
            grad[i][j] .resize (Npx); 
            for (st k = 0; k < Npx; ++k)
            {
                grad[i][j][k] .resize (Npy); 
            };
        };
        for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            for (st ort_2 = 0; ort_2 < 3; ++ort_2)
        {
                    std::cout << ort_1 << " " << ort_2 << std::endl;
            for (st i = 0; i < Npx-1; ++i)
            {
                for (st j = 0; j < Npy-1; ++j)
                {
                    cdbl coor_x = Ri + dx * i;
                    cdbl coor_z = dy * j;

                    cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                    cst i_z = st(coor_z * (1 << n_ref));

                    dbl ksi_x = coor_x - Ri;
                    while (ksi_x > bx)
                        ksi_x -= bx;
                    ksi_x /= bx;

                    dbl ksi_z = coor_z;
                    while (ksi_z > by)
                        ksi_z -= by;
                    ksi_z /= by;

                    // grad[0][0][i][j] = stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1,0,0}, x, x, x, ksi_x, ksi_z);
                    // grad[0][0][i][j] = stress_in_cell (i, j, arr<i32, 3>{1,0,0}, x, x, x, coor_x, coor_z);
                    grad[ort_1][ort_2][i][j] =
                        stress_in_cell (arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, x, x, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{1, 0, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, x, y, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, x, z, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 1, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, y, x, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 1, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, y, y, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 1, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, y, z, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, z, x, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 0, 1}, y, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, z, y, coor_x, coor_z) +
                        stress_in_cell (arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        deform_in_macro(i_x, i_z, z, z, coor_x, coor_z);

                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                        // deform_in_macro(i_x, i_z, x).dy(coor_x, coor_z) +
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // deform_in_macro(i_x, i_z, z).dx(coor_x, coor_z) +
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // deform_in_macro(i_x, i_z, z).dy(coor_x, coor_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
                    //     deform_in_macro(i_z, i_x, x).dx(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
                    //     deform_in_macro(i_z, i_x, x).dy(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
                    //     deform_in_macro(i_z, i_x, z).dx(coor_x, coor_z) +
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                    //     deform_in_macro(i_z, i_x, z).dy(coor_x, coor_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);// *
                        // Vxx;// +
                        // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
                        // Vzz;
                    // grad[ort_1][ort_2][i][j] =1.0;
                        // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, x, x, ksi_x, ksi_z);
                        // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z);
                    // grad[ort_1][ort_2][i][j] =
                    //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);
                        // deform_in_macro(i_x, i_z, ort_1).dx(coor_x, coor_z);
                        // deform_in_macro(i_x, i_z, ort_1)(coor_x, coor_z);
                };
            };
        };
        {
        arr<str, 3> ort = {"x", "y", "z"};
        arr<str, 3> aprx = {"0", "1", "2"};
        for (st approx_number = 1; approx_number < 2; ++approx_number)
        {
            for (st i = 0; i < approx_number+1; ++i)
            {
                for (st j = 0; j < approx_number+1; ++j)
                {
                    for (st k = 0; k < approx_number+1; ++k)
                    {
                        if ((i+j+k) == approx_number)
                        {
                            arr<i32, 3> approximation = {i, j, k};
                            for (st nu = 0; nu < 3; ++nu)
                            {
                                for (st alpha = 0; alpha < 3; ++alpha)
                                {
                                for (st beta = 0; beta < 3; ++beta)
                                {
                                    str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+
                                        str("_")+ort[nu]+str("_")+ort[alpha]+str("_")+ort[beta];
                                    {
                                        std::ofstream out ("ring/stress_cell_"+name+".gpd", std::ios::out);
            for (st m = 0; m < Npx-1; ++m)
            {
                for (st n = 0; n < Npy-1; ++n)
                {
                    cdbl coor_x = Ri + dx * n;
                    cdbl coor_z = dy * m;

                    cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                    cst i_z = st(coor_z * (1 << n_ref));

                    dbl ksi_x = coor_x - Ri;
                    while (ksi_x > bx)
                        ksi_x -= bx;
                    ksi_x /= bx;

                    dbl ksi_z = coor_z;
                    while (ksi_z > by)
                        ksi_z -= by;
                    ksi_z /= by;

                    out << coor_x << " " << coor_z << " " << 
                        stress_in_cell (approximation, nu, alpha, beta, ksi_x, ksi_z)
                        << std::endl;
                };
            };
                                        out.close ();
                                    };
                                };
                                };
                            };
                        };
                    };
                };
            };
        };
        };
        {
            arr<str, 3> ort = {"x", "y", "z"};
            for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            {
                for (st ort_2 = 0; ort_2 < 3; ++ort_2)
                {
                    {
                        std::ofstream f(str("ring/stress_macro_") + ort[ort_1] + ort[ort_2] +".gpd", std::ios::out);
                        for (st i = 0; i < Npx-1; ++i)
                        {
                            for (st j = 0; j < Npy-1; ++j)
                            {
                                cdbl coor_x = Ri + dx * i;
                                cdbl coor_z = dy * j;

                                cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                                cst i_z = st(coor_z * (1 << n_ref));
                                f << coor_x << " " << coor_z << " " << 
                                    deform_in_macro(i_x, i_z, ort_1, ort_2, coor_x, coor_z)
                                    << std::endl;
                            };
                        };
                        f.close ();
                    };
                };
            };
        };
        {
            std::ofstream f(str("ring/stress_macro_anal.gpd"), std::ios::out);
            for (st i = 0; i < Npx-1; ++i)
            {
                for (st j = 0; j < Npy-1; ++j)
                {
                    cdbl coor_x = Ri + dx * i;
                    cdbl coor_z = dy * j;

                    f << coor_x << " " << coor_z << " " << 
                        Ri*Ri*(coor_x*coor_x - Ro*Ro) /((Ro*Ro-Ri*Ri)*coor_x*coor_x)
                        << std::endl;
                };
            };
            f.close ();
        };
        {
            arr<str, 3> ort = {"x", "y", "z"};
            for (st ort_1 = 0; ort_1 < 3; ++ort_1)
            {
                for (st ort_2 = 0; ort_2 < 3; ++ort_2)
                {
                    std::ofstream f(str("ring/stress_real_") + ort[ort_1] + ort[ort_2] + ".gpd", std::ios::out);
                    for (st n = 0; n < Npx-1; ++n)
                    {
                        for (st m = 0; m < Npy-1; ++m)
                        {
                            // f << coor_flat_cell[i][j](x) << " " << coor_flat_cell[i][j](z) << " " << grad[0][0][i][j] << std::endl;
                            cdbl coor_x = Ri + dx * n;
                            cdbl coor_z = dy * m;

                            cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
                            cst i_z = st(coor_z * (1 << n_ref));

                            dbl ksi_x = coor_x - Ri;
                            while (ksi_x > bx)
                                ksi_x -= bx;
                            ksi_x /= bx;

                            dbl ksi_z = coor_z;
                            while (ksi_z > by)
                                ksi_z -= by;
                            ksi_z /= by;

                            f << coor_x << " " << coor_z << " " << 
                                grad[ort_1][ort_2][n][m]
                                // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, 1, 1, ksi_x, ksi_z)
                                // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z)
                                << std::endl;
                            // cell_flat_stress[arr<i32,3>{1,0,0}][x][x][x][i][j] << std::endl;
                        };
                    };
                    f.close ();
                };
            };
        };
    };
};

void get_macro_coef(ATools::FourthOrderTensor &C)
{
    {
        ATools::FourthOrderTensor Cxy;
        std::ifstream in ("meta_coef.bin", std::ios::in | std::ios::binary);
        in.read ((char *) &Cxy, sizeof Cxy);
        in.close ();
        arr<arr<dbl,3>,3> A = {
            arr<dbl,3>{1.0, 0.0, 0.0},
            arr<dbl,3>{0.0, 0.0, -1.0},
            arr<dbl,3>{0.0, 1.0, 0.0}
        };
        for (st i = 0; i < 3; ++i)
            for (st j = 0; j < 3; ++j)
                for (st k = 0; k < 3; ++k)
                    for (st l = 0; l < 3; ++l)
                    {
                        C[i][j][k][l] = 0.0;

                        for (st a = 0; a < 3; ++a)
                            for (st b = 0; b < 3; ++b)
                                for (st c = 0; c < 3; ++c)
                                    for (st d = 0; d < 3; ++d)
                                    {
                                        C[i][j][k][l] += 
                                            A[i][a]*A[j][b]*A[k][c]*A[l][d]*Cxy[a][b][c][d];
                                    };
                    };
    };
};

template <cst n_ref>
void get_macro_move(
        arr<dealii::Vector<dbl>, 3> &move, Domain<2> &domain, 
        const dealii::FiniteElement<2> &fe, const str file_name)
{
    enum {x, y, z};

    arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v;
    arr<arr<dealii::Point<2>,(1 << n_ref) + 1>,(1 << n_ref) + 1> coor;

    // Получаем из файла решение в кольце
    {
        std::ifstream in (file_name, std::ios::in | std::ios::binary);
        for (st i = 0; i < (1 << n_ref) + 1; ++i)
            for (st j = 0; j < (1 << n_ref) + 1; ++j)
            {
                in.read ((char *) &v[i][j][x], sizeof(dbl));
                in.read ((char *) &v[i][j][y], sizeof(dbl));
                in.read ((char *) &v[i][j][z], sizeof(dbl));
            };
        in.close ();
    };

    dealii::GridGenerator::hyper_cube(domain.grid);
    domain.grid.refine_global(n_ref);
    // dealii::FE_Q<2> fe(1);
    domain.dof_init (fe);
    move[x] .reinit (domain.dof_handler.n_dofs());
    move[y] .reinit (domain.dof_handler.n_dofs());
    move[z] .reinit (domain.dof_handler.n_dofs());
    // auto dof = domain.dof_handler.locally_owned_dofs();
    // std::cout << dof.index_within_set(31) << std::endl;
    // dof.print();
    // auto ps = domain.grid.get_vertices();
    // for (st i = 0; i < p.size(); ++i)
    // {
    //     cst i_x = st(p[i](x) * (1 << n_ref));
    //     cst i_y = st(p[i](y) * (1 << n_ref));
    //     move[x](i) = v[i_x][i_y][x];
    //     move[y](i) = v[i_x][i_y][y];
    //     move[z](i) = v[i_x][i_y][z];
    // };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                const dealii::Point<2> p = cell -> vertex(i);
                cst i_x = st(p(x) * (1 << n_ref));
                cst i_y = st(p(y) * (1 << n_ref));
                move[x](cell ->vertex_dof_index (i, 0)) = v[i_x][i_y][x];
                move[y](cell ->vertex_dof_index (i, 0)) = v[i_x][i_y][y];
                move[z](cell ->vertex_dof_index (i, 0)) = v[i_x][i_y][z];//-(p(y)*20.0-10.0);//
            };
        };
    };
    // auto grad =
    //     dealii::VectorTools::point_gradient(domain.dof_handler, move[x], dealii::Point<2>(0.5, 0.5));
    // std::cout << grad[0] << std::endl;
    // HCPTools::print_temperature<2>(move[x], domain.dof_handler, "ring/macro_move_x.gpd");
    // HCPTools::print_temperature<2>(move[y], domain.dof_handler, "ring/macro_move_y.gpd");
    // HCPTools::print_temperature<2>(move[z], domain.dof_handler, "ring/macro_move_z.gpd");
};

void get_macro_deform(
        arr<dealii::Vector<dbl>, 3> &move, Domain<2> &domain, 
        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform, cdbl Ri, cdbl W)
{
    enum {x, y, z};
    // auto v = domain.grid.get_vertices();
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                const dealii::Point<2> p = cell -> vertex(i);
                cst indx = cell ->vertex_dof_index (i, 0);
                cdbl r = p(x) * W + Ri;
                for (st j = 0; j < 3; ++j)
                {
                    // dealii::Tensor<1, 2, dbl> grad =
                    auto grad =
                        dealii::VectorTools::point_gradient(domain.dof_handler, move[j], p);
                    deform[j][x](indx) = grad[x];
                    deform[j][z](indx) = grad[y];
                    // for (st j = 0; j < 2; ++j)
                    // {
                    //     deform[i][j](indx) = grad[j];
                    // };
                };
                deform[y][y](indx) = move[x](indx) / r;
            };
        };
    };
    // arr<str, 3> ort = {"x", "y", "z"};
    // for (st i = 0; i < 3; ++i)
    // {
    //     for (st j = 0; j < 2; ++j)
    //     {
    //         HCPTools::print_temperature<2>(deform[i][j], domain.dof_handler, 
    //                 str("ring/macro_deform_") + ort[i] + ort[j] + "gpd");
    //     };
    // };
};

void get_macro_deform_2(
        arr<dealii::Vector<dbl>, 3> &move, Domain<2> &domain, 
        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform, cdbl H, cdbl W, cdbl Ri)
{
    enum {x, y, z};
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };
    std::cout << "W=" << " " << W << std::endl;
    // {
    //     auto cell = domain.dof_handler.begin_active();
    //     auto endc = domain.dof_handler.end();
    //     for (; cell != endc; ++cell)
    //     {
    //         if ((std::abs(cell->vertex(0)(x) - 0.5) < 1.0e-8) and (std::abs(cell->vertex(0)(y) - 0.5) < 1.0e-8))
    //         {
    //         cdbl px1 = cell->vertex(0)(x)*W+Ri;
    //         cdbl px2 = cell->vertex(1)(x)*W+Ri;
    //         cdbl py1 = cell->vertex(0)(y);
    //         cdbl py2 = cell->vertex(3)(y);
    //         arr<prmt::Point<2>, 4> points = {
    //             prmt::Point<2>(px1, py1),
    //             prmt::Point<2>(px2, py1),
    //             prmt::Point<2>(px2, py2),
    //             prmt::Point<2>(px1, py2)
    //         };
    //             
    //         };
    //     };
    // };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            cdbl px1 = cell->vertex(0)(x)*W+Ri;
            cdbl px2 = cell->vertex(1)(x)*W+Ri;
            cdbl py1 = cell->vertex(0)(y)*H;
            cdbl py2 = cell->vertex(3)(y)*H;
            arr<prmt::Point<2>, 4> points = {
                prmt::Point<2>(px1, py1),
                prmt::Point<2>(px2, py1),
                prmt::Point<2>(px2, py2),
                prmt::Point<2>(px1, py2)
                    // cell->vertex(0),
                    // cell->vertex(1),
                    // cell->vertex(3),
                    // cell->vertex(2)
            };
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                cst indx = cell ->vertex_dof_index (i, 0);
                dealii::Point<2> p = cell -> vertex(i);
                p(x) = p(x) * W + Ri;
                p(y) = p(y) * H;
                // cdbl r = p(x) * W + Ri;
                for (st j = 0; j < 3; ++j)
                {
                    arr<dbl, 4> values = {
                        move[j](cell ->vertex_dof_index (0, 0)),
                        move[j](cell ->vertex_dof_index (1, 0)),
                        move[j](cell ->vertex_dof_index (3, 0)),
                        move[j](cell ->vertex_dof_index (2, 0))
                    };
                    Scalar4PointsFunc<2> func(points, values);

                    deform[j][x](indx) = func.dx(p(x), p(y));
                    deform[j][z](indx) = func.dy(p(x), p(y));
                };
                deform[y][y](indx) = move[x](indx) / p(x);
            };
        };
    };
};

void get_macro_deform_3(
        arr<dealii::Vector<dbl>, 3> &move, Domain<2> &domain, 
        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform, cdbl H, cdbl W, cdbl Ri)
{
    enum {x, y, z};
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };
    {
        cdbl Ro = Ri + W;
        cdbl C1 = 0.75 * Ri*Ri / (Ro*Ro-Ri*Ri);
        cdbl C2 = 1.25 * Ri*Ri * Ro*Ro / (Ro*Ro-Ri*Ri);
        cdbl C3 = -0.5 * Ri*Ri / (Ro*Ro-Ri*Ri);
        std::cout << C1 << " " << C2 << " " << C3 << std::endl;
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                const dealii::Point<2> p = cell -> vertex(i);
                cst indx = cell ->vertex_dof_index (i, 0);
                cdbl r = p(x) * W + Ri;
                deform[x][x](indx) = C1 - C2 / (r*r);
                deform[y][y](indx) = C1 + C2 / (r*r);
                deform[z][z](indx) = C3;
            };
        };
    };
};

void get_micro_deform_stress (
        cst ort_slice,
        cdbl coor_slice,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &stress,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &deform,
        Domain<2> &domain,
        cst number_of_approx,
        const dealii::FiniteElement<2> &fe,
        cdbl R,
        cst n_p
        )
{
    enum {x, y, z};

    OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_stress(3);
    OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_flat_deform(3);
    vec<dealii::Point<3>> coor_flat_cell;

    // Получаем из файла решение для ячейки
    get_flat_deform_and_stress (
            ort_slice,
            coor_slice,
            cell_flat_stress,
            cell_flat_deform,
            coor_flat_cell,
            number_of_approx
            );
    puts("!!!!!!!!!!!!!!!!!!!!!1");
    
    // Строим треангуляцию
    {
        vec<prmt::Point<2>> border;
        vec<st> type_border;
        give_rectangle_with_border_condition(
                border,
                type_border,
                arr<st, 4>{1,3,2,4},
                10,
                prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
        vec<vec<prmt::Point<2>>> inclusion(1);
        dbl radius = R;
        st n_p_on_inc = n_p;
        // {
        //     std::ifstream in ("ring/radius_include.bin", std::ios::in | std::ios::binary);
        //     in.read ((char *) &radius, sizeof(dbl));
        //     in.close ();
        // };
        // {
        //     std::ifstream in ("ring/n_points_on_include.bin", std::ios::in | std::ios::binary);
        //     in.read ((char *) &n_p_on_inc, sizeof(st));
        //     in.close ();
        // };
        dealii::Point<2> center (0.5, 0.5);
        give_circ(inclusion[0], n_p_on_inc, radius, prmt::Point<2>(center));
        ::set_grid(domain.grid, border, inclusion, type_border);
    };
    domain.dof_init (fe);
    puts("!!!!!!!!!!!!!!!!!!!!!2");

    // for (auto &&a : stress.content)
    //     for (auto &&b : a)
    //         for (auto &&c : b)
    //             for (auto &&d : c)
    //                 for (auto &&e : d)
    //                     for (auto &&f : e)
    //                         f .reinit (domain.dof_handler .n_dofs());

    approx_iteration (number_of_approx, 
            [&stress, &deform, &domain]
            (arr<i32, 3> a, cst nu, cst alpha){
            for (st beta = 0; beta < 3; ++beta)
            {
            stress[a][nu][alpha][beta] .reinit (domain.dof_handler .n_dofs());     
            deform[a][nu][alpha][beta] .reinit (domain.dof_handler .n_dofs());     
            };
            });

    // for (auto &&a : deform.content)
    //     for (auto &&b : a)
    //         for (auto &&c : b)
    //             for (auto &&d : c)
    //                 for (auto &&e : d)
    //                     for (auto &&f : e)
    //                         f .reinit (domain.dof_handler .n_dofs());
    puts("!!!!!!!!!!!!!!!!!!!!!3");

    // Распихивание имеющихся напряжений и деформаций по треангуляции
    {
        // approx_iteration (number_of_approx, 
        //         [&stress, &deform, &cell_flat_stress, &cell_flat_deform]
        //         (arr<i32, 3> a, cst nu, cst alpha){
        //         for (st beta = 0; beta < 3; ++beta)
        //         {
        //         for (st j = 0; j < stress[a][nu][alpha][beta].size(); ++j)
        //         stress[a][nu][alpha][beta](j) = cell_flat_stress[a][nu][alpha][beta][j];
        //         // deform[a][nu][alpha][beta](indx) = cell_flat_deform[a][nu][alpha][beta][j];
        //         };
        //         });
// {
    // for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    // {
    //     for (st i = 0; i < approx_number+1; ++i)
    //     {
    //         for (st j = 0; j < approx_number+1; ++j)
    //         {
    //             for (st k = 0; k < approx_number+1; ++k)
    //             {
    //                 if ((i+j+k) == approx_number)
    //                 {
    //                     arr<i32, 3> a = {i, j, k};
    //                     // arr<i32, 3> a = {1, 0, 0};
    //                     for (st nu = 0; nu < 3; ++nu)
    //                     {
    //                         for (st alpha = 0; alpha < 3; ++alpha)
    //                         {
    //             for (st beta = 0; beta < 3; ++beta)
    //             {
    //             for (st n = 0; n < stress[a][nu][alpha][beta].size(); ++n)
    //             // stress[a][nu][alpha][beta](n) = cell_flat_stress[a][nu][alpha][beta][n];
    //             stress[arr<i32,3>{i,j,k}][nu][alpha][beta](n) = 
    //                 cell_flat_stress[arr<i32,3>{i,j,k}][nu][alpha][beta][n];
    //             // deform[a][nu][alpha][beta](indx) = cell_flat_deform[a][nu][alpha][beta][j];
    //             };
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    // for (st nu = 0; nu < 3; ++nu)
    // {
    //     for (st alpha = 0; alpha < 3; ++alpha)
    //     {
    //         for (st beta = 0; beta < 3; ++beta)
            // {
            //     cst nu = y;
            //     cst alpha = x;
            //     cst beta = x;
            //     for (st n = 0; n < stress[arr<i32,3>{1,0,0}][0][0][0].size(); ++n)
            //     {
            //         stress[arr<i32,3>{1,0,0}][nu][alpha][beta](n) = cell_flat_stress[arr<i32,3>{1,0,0}][nu][alpha][beta][n];
            //         stress[arr<i32,3>{0,1,0}][nu][alpha][beta](n) = 1.0;//cell_flat_stress[arr<i32,3>{0,1,0}][nu][alpha][beta][n];
            //         // stress[arr<i32,3>{0,1,0}][y][x][x](n) = cell_flat_stress[arr<i32,3>{0,1,0}][nu][alpha][beta][n];
            //         // stress[arr<i32,3>{0,1,0}][y][x][x](n) = cell_flat_stress[arr<i32,3>{0,1,0}][y][x][x][n];
            //         stress[arr<i32,3>{0,0,1}][nu][alpha][beta](n) = cell_flat_stress[arr<i32,3>{0,0,1}][nu][alpha][beta][n];
            //     };
            //     // if ((nu==y) and (alpha==x) and (beta==x)) break;
            // };
//         };
//     };
// };
        // for (st j = 0; j < stress[arr<i32,3>{0,1,0}][y][x][x].size(); ++j)
        //     stress[arr<i32,3>{0,1,0}][y][x][x](j) = cell_flat_stress[arr<i32,3>{0,1,0}][y][x][x][j];
        arr<st, 3> swap = {x, z, y};
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st i = 0; i < dealii::GeometryInfo<2>::vertices_per_cell; ++i)
            {
                const dealii::Point<2> p = cell -> vertex(i);
                cst indx = cell ->vertex_dof_index(i, 0);
                // approx_iteration (number_of_approx, 
                //         [&stress, &deform, &cell_flat_stress, &cell_flat_deform, indx]
                //         (arr<i32, 3> a, cst nu, cst alpha){
                //         for (st beta = 0; beta < 3; ++beta)
                //         {
                //         stress[a][nu][alpha][beta](indx) = cell_flat_stress[a][nu][alpha][beta][indx];
                //         // deform[a][nu][alpha][beta](indx) = cell_flat_deform[a][nu][alpha][beta][j];
                //         };
                //         });

                for (st j = 0; j < coor_flat_cell.size(); ++j)
                {
                    if (
                            (std::abs(p(x) - coor_flat_cell[j](x)) < 1.0e-10) and
                            (std::abs(p(y) - coor_flat_cell[j](y)) < 1.0e-10)
                       )
                    {

                        // stress[arr<i32,3>{1,0,0}][0][0][0](0) = 0.0;//cell_flat_stress[a][nu][alpha][beta][j];
                        // approx_iteration (number_of_approx, 
                        //         [&stress, &deform, &cell_flat_stress, &cell_flat_deform, indx, j]
                        //         (arr<i32, 3> a, cst nu, cst alpha){
                        //         for (st beta = 0; beta < 3; ++beta)
                        //         {
                        //         stress[a][nu][alpha][beta](j) = cell_flat_stress[a][nu][alpha][beta][j];
                        //         deform[a][nu][alpha][beta](indx) = cell_flat_deform[a][nu][alpha][beta][j];
                        //         };
                        //         });
                        for (st nu = 0; nu < 3; ++nu)
                        {
                            for (st alpha = 0; alpha < 3; ++alpha)
                            {
                                for (st beta = 0; beta < 3; ++beta)
                                {
                        stress[arr<i32, 3>{1,0,0}][nu][alpha][beta](j) = 
                            cell_flat_stress[arr<i32, 3>{1,0,0}][swap[nu]][swap[alpha]][swap[beta]][j];
                        stress[arr<i32, 3>{0,1,0}][nu][alpha][beta](j) = 
                            cell_flat_stress[arr<i32, 3>{0,0,1}][swap[nu]][swap[alpha]][swap[beta]][j];
                        stress[arr<i32, 3>{0,0,1}][nu][alpha][beta](j) = 
                            cell_flat_stress[arr<i32, 3>{0,1,0}][swap[nu]][swap[alpha]][swap[beta]][j];
                                };
                            };
                        };
                        for (st nu = 0; nu < 3; ++nu)
                        {
                            for (st alpha = 0; alpha < 3; ++alpha)
                            {
                                for (st beta = 0; beta < 3; ++beta)
                                {
                        deform[arr<i32, 3>{1,0,0}][nu][alpha][beta](j) = 
                            cell_flat_deform[arr<i32, 3>{1,0,0}][swap[nu]][swap[alpha]][swap[beta]][j];
                        deform[arr<i32, 3>{0,1,0}][nu][alpha][beta](j) = 
                            cell_flat_deform[arr<i32, 3>{0,0,1}][swap[nu]][swap[alpha]][swap[beta]][j];
                        deform[arr<i32, 3>{0,0,1}][nu][alpha][beta](j) = 
                            cell_flat_deform[arr<i32, 3>{0,1,0}][swap[nu]][swap[alpha]][swap[beta]][j];
                                };
                            };
                        };
                        break;
                    };
                };
            };
        };
    };
    // {
    //     std::ofstream f("tmp/stress_yyxx.gpd", std::ios::out);
    //     for (st i = 0; i < coor_flat_cell.size(); ++i)
    //     {
    //        f << coor_flat_cell[i](x) << " " << coor_flat_cell[i](y) << " " << coor_flat_cell[i](z) << " " << 
    //                     cell_flat_stress[arr<i32,3>{0,1,0}][y][x][x][i] << std::endl;
    //     };
    //     f.close ();
    // };
    // HCPTools::print_temperature<2>(
    //         stress[arr<i32,3>{0,1,0}][y][x][x], domain.dof_handler, str("tmp/micro_stress_yyxx.gpd"));
    // puts("!!!!!!!!!!!!!!!!!!!!!4");
};

template <cst n_ref>
void get_real_move_and_stress(
        arr<dealii::Vector<dbl>, 3> &move_macro_a,
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> &move_micro_a,
        arr<dealii::Vector<dbl>, 3> &move,

        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform_macro_a,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &deform_micro_a,
        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform,

        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &stress_micro_a,
        arr<arr<dealii::Vector<dbl>, 3>, 3> &stress, 

        Domain<2> &domain_macro, 
        Domain<2> &domain_micro, 
        Domain<2> &domain, 

        const dealii::FiniteElement<2> &fe,
        cst Ncx, cst Ncy)
{
    enum {x, y, z};

    cdbl bx = 1.0 / Ncx;
    cdbl by = 1.0 / Ncy;

    puts("!!!!!!11");
    dealii::GridGenerator::hyper_cube(domain.grid);
    domain.grid.refine_global(n_ref);
    domain.dof_init (fe);
    move[x] .reinit (domain.dof_handler.n_dofs());
    move[y] .reinit (domain.dof_handler.n_dofs());
    move[z] .reinit (domain.dof_handler.n_dofs());
    puts("!!!!!!11");

    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            stress[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };

    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };
    puts("!!!!!!12");

    auto stress_micro = [&stress_micro_a, &domain_micro] 
        (arr<i32, 3> k, cst nu, cst alpha, cst beta, const dealii::Point<2> &p){
            return dealii::VectorTools::point_value(
                    domain_micro.dof_handler, stress_micro_a[k][nu][alpha][beta], p);
        };

    auto deform_micro = [&deform_micro_a, &domain_micro] 
        (arr<i32, 3> k, cst nu, cst alpha, cst beta, const dealii::Point<2> &p){
            return dealii::VectorTools::point_value(
                    domain_micro.dof_handler, deform_micro_a[k][nu][alpha][beta], p);
        };
    puts("!!!!!!13");

    auto deform_macro = [&deform_macro_a, &domain_macro] 
        (cst i, cst j, const dealii::Point<2> &p){
            return dealii::VectorTools::point_value(
                    domain_macro.dof_handler, deform_macro_a[i][j], p);
            // return dealii::VectorTools::point_gradient(domain.dof_handler, move_macro_a[i], p)[j];
            // if
            //     const dealii::Point<2> p = cell -> vertex(i);
            //     cst indx = cell ->vertex_dof_index (i, 0);
            //     cdbl r = p(x) * W + Ri;
            //     deform[y][y](indx) = move[x](indx) / r;
        };

    arr<arr<dealii::Vector<dbl>, 3>, 3> deform_macro_approx;
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform_macro_approx[i][j] .reinit (domain.dof_handler.n_dofs());
        };
    };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            // arr<prmt::Point<2>, 4> points = {
            //         cell->vertex(0),
            //         cell->vertex(1),
            //         cell->vertex(3),
            //         cell->vertex(2)
            // };
            for (st n = 0; n < dealii::GeometryInfo<2>::vertices_per_cell; ++n)
            {
                const dealii::Point<2> p = cell -> vertex(n);
                cst indx = cell ->vertex_dof_index(n, 0);
                for (st i = 0; i < 3; ++i)
                {
                    for (st j = 0; j < 3; ++j)
                    {
                        // arr<dbl, 4> values = {
                        //     deform_macro_a[i][j](cell ->vertex_dof_index (0, 0)),
                        //     deform_macro_a[i][j](cell ->vertex_dof_index (1, 0)),
                        //     deform_macro_a[i][j](cell ->vertex_dof_index (3, 0)),
                        //     deform_macro_a[i][j](cell ->vertex_dof_index (2, 0))
                        // };
                        // Scalar4PointsFunc<2> func(points, values);
                        // deform_macro_approx[i][j](indx) = func(p);
                        // deform_macro_approx[i][j](indx) = deform_macro_a[i][j](indx);
                        deform_macro_approx[i][j](indx) = dealii::VectorTools::point_value(
                                domain_macro.dof_handler, deform_macro_a[i][j], p);
                    };
                };
            };
        };
    };
    {
        auto cell = domain.dof_handler.begin_active();
        auto endc = domain.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st n = 0; n < dealii::GeometryInfo<2>::vertices_per_cell; ++n)
            // for (st n = 0; n< 16641;++n)
            // cst n = 0;
            {
                const dealii::Point<2> p = cell -> vertex(n);
                cst indx = cell ->vertex_dof_index(n, 0);
                // const dealii::Point<2> p(0.1, 0.25);
                // cst indx = 0;;
                dealii::Point<2> p_ksi = p;
                for (st i = 0; i < Ncx; ++i)
                {
                    if (p_ksi(x) < bx) break;
                    p_ksi(x) -= bx;
                };
                p_ksi(x) /= bx;
                for (st i = 0; i < Ncy; ++i)
                {
                    if (p_ksi(y) < by) break;
                    p_ksi(y) -= by;
                };
                p_ksi(y) /= by;
                // std::cout << p << " " << p_ksi << std::endl;
                for (st i = 0; i < 3; ++i)
                {
                    for (st j = 0; j < 3; ++j)
                    {
                        stress[i][j](indx) =
                            stress_micro(arr<i32, 3>{1, 0, 0}, x, i, j, p_ksi) * deform_macro_approx[x][x](indx) +
                            // stress_micro(arr<i32, 3>{1, 0, 0}, y, i, j, p_ksi) * deform_macro_approx[x][y](indx) +
                            // stress_micro(arr<i32, 3>{1, 0, 0}, z, i, j, p_ksi) * deform_macro_approx[x][z](indx) +
                            // stress_micro(arr<i32, 3>{0, 1, 0}, x, i, j, p_ksi) * deform_macro_approx[y][x](indx) +
                            stress_micro(arr<i32, 3>{0, 1, 0}, y, i, j, p_ksi) * deform_macro_approx[y][y](indx) +
                            // stress_micro(arr<i32, 3>{0, 1, 0}, z, i, j, p_ksi) * deform_macro_approx[y][z](indx) +
                            // stress_micro(arr<i32, 3>{0, 0, 1}, x, i, j, p_ksi) * deform_macro_approx[z][x](indx) +
                            // stress_micro(arr<i32, 3>{0, 0, 1}, y, i, j, p_ksi) * deform_macro_approx[z][y](indx) +
                            stress_micro(arr<i32, 3>{0, 0, 1}, z, i, j, p_ksi) * deform_macro_approx[z][z](indx)
                            ;
                        deform[i][j](indx) =
                            deform_micro(arr<i32, 3>{1, 0, 0}, x, i, j, p_ksi) * deform_macro_approx[x][x](indx) +
                            // deform_micro(arr<i32, 3>{1, 0, 0}, y, i, j, p_ksi) * deform_macro_approx[x][y](indx) +
                            // deform_micro(arr<i32, 3>{1, 0, 0}, z, i, j, p_ksi) * deform_macro_approx[x][z](indx) +
                            // deform_micro(arr<i32, 3>{0, 1, 0}, x, i, j, p_ksi) * deform_macro_approx[y][x](indx) +
                            deform_micro(arr<i32, 3>{0, 1, 0}, y, i, j, p_ksi) * deform_macro_approx[y][y](indx) +
                            // deform_micro(arr<i32, 3>{0, 1, 0}, z, i, j, p_ksi) * deform_macro_approx[y][z](indx) +
                            // deform_micro(arr<i32, 3>{0, 0, 1}, x, i, j, p_ksi) * deform_macro_approx[z][x](indx) +
                            // deform_micro(arr<i32, 3>{0, 0, 1}, y, i, j, p_ksi) * deform_macro_approx[z][y](indx) +
                            deform_micro(arr<i32, 3>{0, 0, 1}, z, i, j, p_ksi) * deform_macro_approx[z][z](indx)
                            ;
                    };
                };
            };
        };
    };
    str name = "W/stress_line_";
    cst N = 500;
    {
        std::ofstream f(name+"zz_1.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.1);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, z, z, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, z, z, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, z, z, p_ksi) * deform_macro(z, z, p);
            // std::cout << p(x) << std::endl;
            // f << p(x) << std::endl;
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };
    {
        std::ofstream f(name+"yy_1.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.1);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, y, y, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, y, y, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, y, y, p_ksi) * deform_macro(z, z, p);
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };
    {
        std::ofstream f(name+"zz_2.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.2);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, z, z, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, z, z, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, z, z, p_ksi) * deform_macro(z, z, p);
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };
    {
        std::ofstream f(name+"yy_2.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.2);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, y, y, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, y, y, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, y, y, p_ksi) * deform_macro(z, z, p);
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };
    {
        std::ofstream f(name+"xz_1.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.125);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, x, z, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, x, z, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, x, z, p_ksi) * deform_macro(z, z, p);
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };
    {
        std::ofstream f(name+"xz_2.gpd", std::ios::out);
        for (st n = 0; n < N; ++n)
        {
            const dealii::Point<2> p(n*(1.0/N), 0.15);
            dealii::Point<2> p_ksi = p;
            for (st i = 0; i < Ncx; ++i)
            {
                if (p_ksi(x) < bx) break;
                p_ksi(x) -= bx;
            };
            p_ksi(x) /= bx;
            for (st i = 0; i < Ncy; ++i)
            {
                if (p_ksi(y) < by) break;
                p_ksi(y) -= by;
            };
            p_ksi(y) /= by;
            cdbl stress = 
                stress_micro(arr<i32, 3>{1, 0, 0}, x, x, z, p_ksi) * deform_macro(x, x, p) +
                stress_micro(arr<i32, 3>{0, 1, 0}, y, x, z, p_ksi) * deform_macro(y, y, p) +
                stress_micro(arr<i32, 3>{0, 0, 1}, z, x, z, p_ksi) * deform_macro(z, z, p);
            f << p(x) << " " << stress << std::endl;
        };
        f.close ();
    };





    // arr<str, 3> ort = {"x", "y", "z"};
    // cdbl y_1 = 0.1;
    // cdbl y_2 = 0.2;
    // dbl ksi_y_1 = y_1;
    // for (st i = 0; i < Ncy; ++i)
    // {
    //     if (ksi_y_1 < by) break;
    //     ksi_y_1 -= by;
    // };
    // ksi_y_1 /= by;
    // std::cout << ksi_y_1 << std::endl;
    // dbl ksi_y_2 = y_2;
    // for (st i = 0; i < Ncy; ++i)
    // {
    //     if (ksi_y_2 < by) break;
    //     ksi_y_2 -= by;
    // };
    // ksi_y_2 /= by;
    // cst N = 100;
    // for (st m = 0; m < 3; ++m)
    // {
    //     for (st n = 0; n < 3; ++n)
    //     {
    //         std::ofstream f(str("W/stress_line_") + ort[m] + ort[n] + ".gpd", std::ios::out);
    //         for (st i = 0; i < N+1; ++i)
    //         {
    //             dbl stress_1; 
    //             dbl stress_2; 
    //             cdbl X = i*(1.0/N);
    //             {
    //                 const dealii::Point<2> p(X, y_1);
    //                 dealii::Point<2> p_ksi(p(x), ksi_y_1);
    //                 for (st i = 0; i < Ncx; ++i)
    //                 {
    //                     if (p_ksi(x) < bx) break;
    //                     p_ksi(x) -= bx;
    //                 };
    //                 stress_1 = 
    //             stress_micro(arr<i32, 3>{1, 0, 0}, x, y, y, p_ksi) * deform_macro(x, x, p) +
    //             stress_micro(arr<i32, 3>{0, 1, 0}, y, y, y, p_ksi) * deform_macro(y, y, p) +
    //             stress_micro(arr<i32, 3>{0, 0, 1}, z, y, y, p_ksi) * deform_macro(z, z, p);
    //                     // stress_micro(arr<i32, 3>{1, 0, 0}, x, m, n, p_ksi) * deform_macro(x, x, p) +
    //                     // stress_micro(arr<i32, 3>{0, 1, 0}, y, m, n, p_ksi) * deform_macro(y, y, p) +
    //                     // stress_micro(arr<i32, 3>{0, 0, 1}, z, m, n, p_ksi) * deform_macro(z, z, p);
    //             };
    //             {
    //                 const dealii::Point<2> p(X, y_2);
    //                 dealii::Point<2> p_ksi(p(x), ksi_y_2);
    //                 for (st i = 0; i < Ncx; ++i)
    //                 {
    //                     if (p_ksi(x) < bx) break;
    //                     p_ksi(x) -= bx;
    //                 };
    //                 stress_2 = 
    //                     stress_micro(arr<i32, 3>{1, 0, 0}, x, m, n, p_ksi) * deform_macro(x, x, p) +
    //                     stress_micro(arr<i32, 3>{0, 1, 0}, y, m, n, p_ksi) * deform_macro(y, y, p) +
    //                     stress_micro(arr<i32, 3>{0, 0, 1}, z, m, n, p_ksi) * deform_macro(z, z, p);
    //             };
    //             f << X << " " << stress_1 << " " << stress_2 << std::endl;
    //         };
    //         f.close ();
    //     };
    // };
    puts("!!!!!!14");
};

template <cst n_ref>
void get_real_move_and_stress(
        arr<dealii::Vector<dbl>, 3> &move_macro_a,
        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> &move_micro_a,
        arr<dealii::Vector<dbl>, 3> &move,

        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform_macro_a,
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &deform_micro_a,
        arr<arr<dealii::Vector<dbl>, 3>, 3> &deform,

        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> &stress_micro_a,
        arr<arr<dealii::Vector<dbl>, 3>, 3> &stress, 

        Domain<2> &domain_macro, 
        Domain<2> &domain_micro, 
        Domain<2> &domain, 

        const dealii::FiniteElement<2> &fe,
        cst Ncx, cst Ncy, const prmt::Point<2> &center)
{
    enum {x, y, z};

    cdbl bx = 1.0 / Ncx;
    cdbl by = 1.0 / Ncy;

    move[x] .reinit (domain_micro.dof_handler.n_dofs());
    move[y] .reinit (domain_micro.dof_handler.n_dofs());
    move[z] .reinit (domain_micro.dof_handler.n_dofs());
    puts("!!!!!!11");

    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            stress[i][j] .reinit (domain_micro.dof_handler.n_dofs());
            deform[i][j] .reinit (domain_micro.dof_handler.n_dofs());
        };
    };

    arr<arr<dealii::Vector<dbl>, 3>, 3> deform_macro_approx;
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            deform_macro_approx[i][j] .reinit (domain_micro.dof_handler.n_dofs());
        };
    };
    {
        auto cell = domain_micro.dof_handler.begin_active();
        auto endc = domain_micro.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st n = 0; n < dealii::GeometryInfo<2>::vertices_per_cell; ++n)
            {
                dbl X = cell ->vertex(n)(x);
                dbl Y = cell ->vertex(n)(y);
                X = (X - 0.5) * bx + center.x();
                Y = (Y - 0.5) * by + center.y();
                const dealii::Point<2> p(X, Y);
                cst indx = cell ->vertex_dof_index(n, 0);
                for (st i = 0; i < 3; ++i)
                {
                    for (st j = 0; j < 3; ++j)
                    {
                        deform_macro_approx[i][j](indx) = dealii::VectorTools::point_value(
                                domain_macro.dof_handler, deform_macro_a[i][j], p);
                    };
                };
            };
        };
    };
    {
        auto cell = domain_micro.dof_handler.begin_active();
        auto endc = domain_micro.dof_handler.end();
        for (; cell != endc; ++cell)
        {
            for (st n = 0; n < dealii::GeometryInfo<2>::vertices_per_cell; ++n)
            {
                const dealii::Point<2> p = cell -> vertex(n);
                cst indx = cell ->vertex_dof_index(n, 0);
                for (st i = 0; i < 3; ++i)
                {
                    for (st j = 0; j < 3; ++j)
                    {
                        stress[i][j](indx) =
                            stress_micro_a[arr<i32, 3>{1, 0, 0}][x][i][j](indx) * deform_macro_approx[x][x](indx) +
                            stress_micro_a[arr<i32, 3>{1, 0, 0}][y][i][j](indx) * deform_macro_approx[x][y](indx) +
                            stress_micro_a[arr<i32, 3>{1, 0, 0}][z][i][j](indx) * deform_macro_approx[x][z](indx) +
                            stress_micro_a[arr<i32, 3>{0, 1, 0}][x][i][j](indx) * deform_macro_approx[y][x](indx) +
                            stress_micro_a[arr<i32, 3>{0, 1, 0}][y][i][j](indx) * deform_macro_approx[y][y](indx) +
                            stress_micro_a[arr<i32, 3>{0, 1, 0}][z][i][j](indx) * deform_macro_approx[y][z](indx) +
                            stress_micro_a[arr<i32, 3>{0, 0, 1}][x][i][j](indx) * deform_macro_approx[z][x](indx) +
                            stress_micro_a[arr<i32, 3>{0, 0, 1}][y][i][j](indx) * deform_macro_approx[z][y](indx) +
                            stress_micro_a[arr<i32, 3>{0, 0, 1}][z][i][j](indx) * deform_macro_approx[z][z](indx);
                        deform[i][j](indx) =
                            deform_micro_a[arr<i32, 3>{1, 0, 0}][x][i][j](indx) * deform_macro_approx[x][x](indx) +
                            deform_micro_a[arr<i32, 3>{1, 0, 0}][y][i][j](indx) * deform_macro_approx[x][y](indx) +
                            deform_micro_a[arr<i32, 3>{1, 0, 0}][z][i][j](indx) * deform_macro_approx[x][z](indx) +
                            deform_micro_a[arr<i32, 3>{0, 1, 0}][x][i][j](indx) * deform_macro_approx[y][x](indx) +
                            deform_micro_a[arr<i32, 3>{0, 1, 0}][y][i][j](indx) * deform_macro_approx[y][y](indx) +
                            deform_micro_a[arr<i32, 3>{0, 1, 0}][z][i][j](indx) * deform_macro_approx[y][z](indx) +
                            deform_micro_a[arr<i32, 3>{0, 0, 1}][x][i][j](indx) * deform_macro_approx[z][x](indx) +
                            deform_micro_a[arr<i32, 3>{0, 0, 1}][y][i][j](indx) * deform_macro_approx[z][y](indx) +
                            deform_micro_a[arr<i32, 3>{0, 0, 1}][z][i][j](indx) * deform_macro_approx[z][z](indx);
                            // stress_micro_a[arr<i32, 3>{0, 1, 0}][y][i][j](indx);
                            // deform_macro_approx[y][y](indx);// +
                    };
                };
            };
        };
    };
    puts("!!!!!!14");
};

template <cst n_ref_macro, cst n_ref_real>
void calculate_real_stress_in_ring_arbitrary_grid_alternate(
        cst flag, cdbl H, cdbl W, cdbl Ri, cst Ncx, cst Ncy, cdbl R_fiber, cst n_p)
        // cst flag, cdbl ratio, cdbl Ri, cdbl Ro,
        // cst Ncx, cst Npx, cst Npy)
{
    if (flag)
    {  
        enum {x, y, z};
        arr<str, 3> ort = {"x", "y", "z"};

        dealii::FE_Q<2> fe(1);

        arr<dealii::Vector<dbl>, 3> move_macro;
        Domain<2> domain_macro;

        get_macro_move<n_ref_macro>(move_macro, domain_macro, fe, "ring_move.bin");
        HCPTools::print_temperature<2>(move_macro[x], domain_macro.dof_handler, "ring/macro/move/x.gpd");
        HCPTools::print_temperature<2>(move_macro[y], domain_macro.dof_handler, "ring/macro/move/y.gpd");
        HCPTools::print_temperature<2>(move_macro[z], domain_macro.dof_handler, "ring/macro/move/z.gpd");

        arr<arr<dealii::Vector<dbl>, 3>, 3> deform_macro;
        get_macro_deform_2(move_macro, domain_macro, deform_macro, H, W, Ri);
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                HCPTools::print_temperature<2>(deform_macro[i][j], domain_macro.dof_handler, 
                        str("ring/macro/deform/") + ort[i] + ort[j] + ".gpd");
            };
        };

        arr<arr<dealii::Vector<dbl>, 3>, 3> stress_macro;
        ATools::FourthOrderTensor C;
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.25}(C);
        get_macro_coef(C);
        std::cout << "Czzii " << C[z][z][x][x] << " " << C[z][z][y][y] << " " << C[z][z][z][z] << std::endl;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                stress_macro[i][j] .reinit (domain_macro.dof_handler.n_dofs());
                for (st n = 0; n < domain_macro.dof_handler.n_dofs(); ++n)
                {
                    stress_macro[i][j](n) = 0.0;
                    for (st k = 0; k < 3; ++k)
                    {
                        for (st l = 0; l < 3; ++l)
                        {
                            stress_macro[i][j](n) += C[i][j][k][l] * deform_macro[k][l](n);
                        };
                    };
                };
            };
        };
        puts("C[][][][]");
        std::cout << C[x][x][x][x] << " " << C[x][x][y][y] << " " << C[x][x][z][z] << std::endl;
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                HCPTools::print_temperature<2>(stress_macro[i][j], domain_macro.dof_handler, 
                        str("ring/macro/stress/") + ort[i] + ort[j] + ".gpd");
            };
        };


        OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> move_micro(2);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> stress_micro(2);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> deform_micro(2);
        Domain<2> domain_micro;
        get_micro_deform_stress (z, 0.5, stress_micro, deform_micro, domain_micro, 2, fe, R_fiber, n_p);
        puts("!!!!!!!!!5");
        approx_iteration (2, 
                [&stress_micro, &deform_micro, &domain_micro]
                (arr<i32, 3> a, cst nu, cst alpha){
                arr<str, 3> ort = {"x", "y", "z"};
                arr<str, 3> aprx = {"0", "1", "2"};
                for (st i = 0; i < 3; ++i)
                {
                str name = aprx[a[0]]+str("_")+aprx[a[1]]+str("_")+aprx[a[2]]+str("_")+ort[nu]+str("_")+ort[alpha]+ort[i];
                HCPTools::print_temperature<2>(
                    stress_micro[a][nu][alpha][i], domain_micro.dof_handler, str("ring/micro/stress/") + name +".gpd");
                HCPTools::print_temperature<2>(
                    deform_micro[a][nu][alpha][i], domain_micro.dof_handler, str("ring/micro/deform/") + name +".gpd");
                // HCPTools::print_temperature<2>(
                //     deform_micro[a][nu][alpha][i], domain_micro.dof_handler, str("ring/micro_deform_") + name +".gpd");
                };
                });
        puts("!!!!!!10");


        arr<dealii::Vector<dbl>, 3> move_real;
        arr<arr<dealii::Vector<dbl>, 3>, 3> stress_real;
        arr<arr<dealii::Vector<dbl>, 3>, 3> deform_real;
        Domain<2> domain_real;
        // get_real_move_and_stress<n_ref_real>(
        //         move_macro,   move_micro,   move_real, 
        //         deform_macro, deform_micro, deform_real, 
        //         stress_micro, stress_real, 
        //         domain_macro, domain_micro, domain_real,
        //         fe, Ncx, Ncy, prmt::Point<2>(0.1, 0.1));
        // for (st i = 0; i < 3; ++i)
        // {
        //     for (st j = 0; j < 3; ++j)
        //     {
        //         HCPTools::print_temperature<2>(stress_real[i][j], domain_micro.dof_handler, 
        //                 str("ring/real/stress/") + ort[i] + ort[j] + ".gpd");
        //         HCPTools::print_temperature<2>(deform_real[i][j], domain_micro.dof_handler, 
        //                 str("ring/real/deform/") + ort[i] + ort[j] + ".gpd");
        //     };
        // };
        get_real_move_and_stress<n_ref_real>(
                move_macro,   move_micro,   move_real, 
                deform_macro, deform_micro, deform_real, 
                stress_micro, stress_real, 
                domain_macro, domain_micro, domain_real,
                fe, Ncx, Ncy);
        for (st i = 0; i < 3; ++i)
        {
            for (st j = 0; j < 3; ++j)
            {
                HCPTools::print_temperature<2>(stress_real[i][j], domain_real.dof_handler, 
                        str("ring/real/stress/") + ort[i] + ort[j] + ".gpd");
                HCPTools::print_temperature<2>(deform_real[i][j], domain_real.dof_handler, 
                        str("ring/real/deform/") + ort[i] + ort[j] + ".gpd");
                // HCPTools::print_temperature<2>(deform_macro[i][j], domain_macro.dof_handler, 
                //         str("ring/real_stress_") + ort[i] + ort[j] + ".gpd");
            };
        };
        // HCPTools::print_temperature_slice (stress_real[z][z], 
        //         domain_real.dof_handler,
        //         "stress_line_zz.gpd",
        //         y,
        //         0.2);
        // // for (st i = 0; i < 3; ++i)
        // // {
        // //     for (st j = 0; j < 3; ++j)
        // //     {
        // //         HCPTools::print_temperature<2>(deform_real[i][j], domain_real.dof_handler, 
        // //                 str("ring/real_deform_") + ort[i] + ort[j] + ".gpd");
        // //     };
        // // };
    };
//
//         cdbl width = 1.0 / ratio;
//         cst Ncy = Ncx * ratio;
//         cdbl bx = width / Ncx;
//         cdbl by = 1.0 / Ncy;
//         cdbl dx = width / (Npx-1);
//         cdbl dy = 1.0 / (Npy-1);
//
//         arr<arr<arr<dbl,3>,(1 << n_ref) + 1>,(1 << n_ref) + 1> v;
//         arr<arr<dealii::Point<2>,(1 << n_ref) + 1>,(1 << n_ref) + 1> coor;
//
//         // Получаем из файла решение в кольце
//         {
//             {
//                 std::ifstream in ("ring_coor.bin", std::ios::in | std::ios::binary);
//                 for (st i = 0; i < (1 << n_ref) + 1; ++i)
//                     for (st j = 0; j < (1 << n_ref) + 1; ++j)
//                     {
//                         in.read ((char *) &coor[i][j](x), sizeof(dbl));
//                         in.read ((char *) &coor[i][j](y), sizeof(dbl));
//                     };
//                 in.close ();
//             };
//
//             {
//                 std::ifstream in ("ring_move.bin", std::ios::in | std::ios::binary);
//                 for (st i = 0; i < (1 << n_ref) + 1; ++i)
//                     for (st j = 0; j < (1 << n_ref) + 1; ++j)
//                     {
//                         in.read ((char *) &v[i][j][x], sizeof(dbl));
//                         in.read ((char *) &v[i][j][y], sizeof(dbl));
//                         in.read ((char *) &v[i][j][z], sizeof(dbl));
//                     };
//                 in.close ();
//             };
//         };
//
//         Domain<3> domain_cell;
//         OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> cell_stress(3);
//         OnCell::ArrayWithAccessToVector<arr<arr<arr<dealii::Vector<dbl>, 3>, 3>, 3>> cell_deform(3);
//             get_flat_deform_stress_and_domain (
//                     z,
//                     0.5,
//                     cell_stress,
//                     cell_deform,
//                     2
//                     );
//
//
//         auto stress_in_cell = [&cell_flat_stress, &coor_flat_cell, &grid, &v_in_cell] 
//             (arr<i32, 3> k, cst nu, cst alpha, cst beta, cdbl px, cdbl py){
//                 arr<st, 4> v;
//                 {
//                     auto cell = grid.begin_active();
//                     auto endc = grid.end();
//                     for (; cell != endc; ++cell)
//                     {
//                         if (point_in_cell(dealii::Point<2>(px, py), cell))
//                         {
//                             v[0] = v_in_cell[cell->vertex_index(0)];
//                             v[1] = v_in_cell[cell->vertex_index(1)];
//                             v[2] = v_in_cell[cell->vertex_index(3)];
//                             v[3] = v_in_cell[cell->vertex_index(2)];
//                             break;
//                         };
//                     };
//                 };
//                 arr<prmt::Point<2>, 4> points = {
//                     prmt::Point<2>(coor_flat_cell[v[0]](0), coor_flat_cell[v[0]](1)),
//                     prmt::Point<2>(coor_flat_cell[v[1]](0), coor_flat_cell[v[1]](1)),
//                     prmt::Point<2>(coor_flat_cell[v[2]](0), coor_flat_cell[v[2]](1)),
//                     prmt::Point<2>(coor_flat_cell[v[3]](0), coor_flat_cell[v[3]](1))
//                 };
//                 // std::cout << i << " " << j << " ";
//                 // for (st i = 0; i < 4; ++i)
//                 // {
//                 //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
//                 // };
//                 // std::cout << std::endl;
//                 arr<dbl, 4> values = {
//                     cell_flat_stress[k][nu][alpha][beta][v[0]],
//                     cell_flat_stress[k][nu][alpha][beta][v[1]],
//                     cell_flat_stress[k][nu][alpha][beta][v[2]],
//                     cell_flat_stress[k][nu][alpha][beta][v[3]]};
//
//                 Scalar4PointsFunc<2> func(points, values);
//
//                 // return   cell_flat_stress[k][nu][alpha][beta][i][j];
//                 return func(prmt::Point<2>(px, py));
//             };
//
//         auto move_in_macro = [&v, &coor] 
//             (cst i, cst j, cst alpha, cst beta, cdbl px, cdbl py){
//             arr<prmt::Point<2>, 4> points = {
//                 prmt::Point<2>(coor[i][j](0),     coor[i][j](1)),
//                 prmt::Point<2>(coor[i+1][j](0),   coor[i+1][j](1)),
//                 prmt::Point<2>(coor[i+1][j+1](0), coor[i+1][j+1](1)),
//                 prmt::Point<2>(coor[i][j+1](0),   coor[i][j+1](1))};
//             // std::cout << i << " " << j << " ";
//             // for (st i = 0; i < 4; ++i)
//             // {
//             //     std::cout << "(" << points[i].x() << ", " << points[i].y() << ") ";
//             // };
//             // std::cout << std::endl;
//             arr<dbl, 4> values = {
//                 v[i][j][alpha],
//                 v[i+1][j][alpha],
//                 v[i+1][j+1][alpha],
//                 v[i][j+1][alpha]};
//
//                 Scalar4PointsFunc<2> func(points, values);
//
//                 dbl res = 0.0; 
//                 switch (beta) {
//                     case 0: 
//                         res = func.dx(px, py);
//                         break;
//                     case 1:
//                         res = func(px, py) / std::pow(px*px + py*py, 0.5);
//                         break;
//                     case 2:
//                         res = func.dy(px, py);
//                         break;
//                 };
//                 return res; //func;//.dy(prmt::Point<2>(px, py));
//             };
//
//         arr<arr<vec<vec<dbl>>,3>,3> grad;
//         for (st i = 0; i < 3; ++i)
//         for (st j = 0; j < 3; ++j)
//         {
//             grad[i][j] .resize (Npx); 
//             for (st k = 0; k < Npx; ++k)
//             {
//                 grad[i][j][k] .resize (Npy); 
//             };
//         };
//         for (st ort_1 = 0; ort_1 < 3; ++ort_1)
//             for (st ort_2 = 0; ort_2 < 3; ++ort_2)
//         {
//                     std::cout << ort_1 << " " << ort_2 << std::endl;
//             for (st i = 0; i < Npx-1; ++i)
//             {
//                 for (st j = 0; j < Npy-1; ++j)
//                 {
//                     cdbl coor_x = Ri + dx * i;
//                     cdbl coor_z = dy * j;
//
//                     cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
//                     cst i_z = st(coor_z * (1 << n_ref));
//
//                     dbl ksi_x = coor_x - Ri;
//                     while (ksi_x > bx)
//                         ksi_x -= bx;
//                     ksi_x /= bx;
//
//                     dbl ksi_z = coor_z;
//                     while (ksi_z > by)
//                         ksi_z -= by;
//                     ksi_z /= by;
//
//                     // grad[0][0][i][j] = stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1,0,0}, x, x, x, ksi_x, ksi_z);
//                     // grad[0][0][i][j] = stress_in_cell (i, j, arr<i32, 3>{1,0,0}, x, x, x, coor_x, coor_z);
//                     grad[ort_1][ort_2][i][j] =
//                         stress_in_cell (arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, x, x, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{1, 0, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, x, y, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, x, z, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 1, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, y, x, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 1, 0}, y, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, y, y, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 1, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, y, z, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, z, x, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 0, 1}, y, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, z, y, coor_x, coor_z) +
//                         stress_in_cell (arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         move_in_macro(i_x, i_z, z, z, coor_x, coor_z);
//
//                         // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                         // move_in_macro(i_x, i_z, x).dy(coor_x, coor_z) +
//                         // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         // move_in_macro(i_x, i_z, z).dx(coor_x, coor_z) +
//                         // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         // move_in_macro(i_x, i_z, z).dy(coor_x, coor_z);
//                     // grad[ort_1][ort_2][i][j] =
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                     //     move_in_macro(i_z, i_x, x).dx(coor_x, coor_z) +
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, x, ort_1, ort_2, ksi_x, ksi_z) *
//                     //     move_in_macro(i_z, i_x, x).dy(coor_x, coor_z) +
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                     //     move_in_macro(i_z, i_x, z).dx(coor_x, coor_z) +
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                     //     move_in_macro(i_z, i_x, z).dy(coor_x, coor_z);
//                     // grad[ort_1][ort_2][i][j] =
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);// *
//                         // Vxx;// +
//                         // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z) *
//                         // Vzz;
//                     // grad[ort_1][ort_2][i][j] =1.0;
//                         // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, x, x, ksi_x, ksi_z);
//                         // stress_in_cell (i_ksi_x, i_ksi_z, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z);
//                     // grad[ort_1][ort_2][i][j] =
//                     //     stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{1, 0, 0}, x, ort_1, ort_2, ksi_x, ksi_z);
//                         // move_in_macro(i_x, i_z, ort_1).dx(coor_x, coor_z);
//                         // move_in_macro(i_x, i_z, ort_1)(coor_x, coor_z);
//                 };
//             };
//         };
//         {
//         arr<str, 3> ort = {"x", "y", "z"};
//         arr<str, 3> aprx = {"0", "1", "2"};
//         for (st approx_number = 1; approx_number < 2; ++approx_number)
//         {
//             for (st i = 0; i < approx_number+1; ++i)
//             {
//                 for (st j = 0; j < approx_number+1; ++j)
//                 {
//                     for (st k = 0; k < approx_number+1; ++k)
//                     {
//                         if ((i+j+k) == approx_number)
//                         {
//                             arr<i32, 3> approximation = {i, j, k};
//                             for (st nu = 0; nu < 3; ++nu)
//                             {
//                                 for (st alpha = 0; alpha < 3; ++alpha)
//                                 {
//                                 for (st beta = 0; beta < 3; ++beta)
//                                 {
//                                     str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+
//                                         str("_")+ort[nu]+str("_")+ort[alpha]+str("_")+ort[beta];
//                                     {
//                                         std::ofstream out ("ring/stress_cell_"+name+".gpd", std::ios::out);
//             for (st m = 0; m < Npx-1; ++m)
//             {
//                 for (st n = 0; n < Npy-1; ++n)
//                 {
//                     cdbl coor_x = Ri + dx * n;
//                     cdbl coor_z = dy * m;
//
//                     cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
//                     cst i_z = st(coor_z * (1 << n_ref));
//
//                     dbl ksi_x = coor_x - Ri;
//                     while (ksi_x > bx)
//                         ksi_x -= bx;
//                     ksi_x /= bx;
//
//                     dbl ksi_z = coor_z;
//                     while (ksi_z > by)
//                         ksi_z -= by;
//                     ksi_z /= by;
//
//                     out << coor_x << " " << coor_z << " " << 
//                         stress_in_cell (approximation, nu, alpha, beta, ksi_x, ksi_z)
//                         << std::endl;
//                 };
//             };
//                                         out.close ();
//                                     };
//                                 };
//                                 };
//                             };
//                         };
//                     };
//                 };
//             };
//         };
//         };
//         {
//             arr<str, 3> ort = {"x", "y", "z"};
//             for (st ort_1 = 0; ort_1 < 3; ++ort_1)
//             {
//                 for (st ort_2 = 0; ort_2 < 3; ++ort_2)
//                 {
//                     {
//                         std::ofstream f(str("ring/stress_macro_") + ort[ort_1] + ort[ort_2] +".gpd", std::ios::out);
//                         for (st i = 0; i < Npx-1; ++i)
//                         {
//                             for (st j = 0; j < Npy-1; ++j)
//                             {
//                                 cdbl coor_x = Ri + dx * i;
//                                 cdbl coor_z = dy * j;
//
//                                 cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
//                                 cst i_z = st(coor_z * (1 << n_ref));
//                                 f << coor_x << " " << coor_z << " " << 
//                                     move_in_macro(i_x, i_z, ort_1, ort_2, coor_x, coor_z)
//                                     << std::endl;
//                             };
//                         };
//                         f.close ();
//                     };
//                 };
//             };
//         };
//         {
//             std::ofstream f(str("ring/stress_macro_anal.gpd"), std::ios::out);
//             for (st i = 0; i < Npx-1; ++i)
//             {
//                 for (st j = 0; j < Npy-1; ++j)
//                 {
//                     cdbl coor_x = Ri + dx * i;
//                     cdbl coor_z = dy * j;
//
//                     f << coor_x << " " << coor_z << " " << 
//                         Ri*Ri*(coor_x*coor_x - Ro*Ro) /((Ro*Ro-Ri*Ri)*coor_x*coor_x)
//                         << std::endl;
//                 };
//             };
//             f.close ();
//         };
//         {
//             arr<str, 3> ort = {"x", "y", "z"};
//             for (st ort_1 = 0; ort_1 < 3; ++ort_1)
//             {
//                 for (st ort_2 = 0; ort_2 < 3; ++ort_2)
//                 {
//                     std::ofstream f(str("ring/stress_real_") + ort[ort_1] + ort[ort_2] + ".gpd", std::ios::out);
//                     for (st n = 0; n < Npx-1; ++n)
//                     {
//                         for (st m = 0; m < Npy-1; ++m)
//                         {
//                             // f << coor_flat_cell[i][j](x) << " " << coor_flat_cell[i][j](z) << " " << grad[0][0][i][j] << std::endl;
//                             cdbl coor_x = Ri + dx * n;
//                             cdbl coor_z = dy * m;
//
//                             cst i_x = st(((coor_x - Ri) / width) * (1 << n_ref));
//                             cst i_z = st(coor_z * (1 << n_ref));
//
//                             dbl ksi_x = coor_x - Ri;
//                             while (ksi_x > bx)
//                                 ksi_x -= bx;
//                             ksi_x /= bx;
//
//                             dbl ksi_z = coor_z;
//                             while (ksi_z > by)
//                                 ksi_z -= by;
//                             ksi_z /= by;
//
//                             f << coor_x << " " << coor_z << " " << 
//                                 grad[ort_1][ort_2][n][m]
//                                 // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, 1, 1, ksi_x, ksi_z)
//                                 // stress_in_cell (i_ksi_z, i_ksi_x, arr<i32, 3>{0, 0, 1}, z, ort_1, ort_2, ksi_x, ksi_z)
//                                 << std::endl;
//                             // cell_flat_stress[arr<i32,3>{1,0,0}][x][x][x][i][j] << std::endl;
//                         };
//                     };
//                     f.close ();
//                 };
//             };
//         };
//     };
};

void calculate_real_stress (
        cst flag, cst num_cells, cdbl E, cdbl pua, cst flg1, cst flg2, cdbl R, cst refine_cell, cst refine_hole,
        const str f_name_cell, const str f_name_hole)
{
    if (flag)
    {  
        enum {x, y, z};


        ///////////////////////////////// Задание параметров.

        cst ort_slice = y;
        cdbl coor_slice = 0.5;
        cst number_of_approx = 2;

        // solve_approx_cell_elastic_problem (flg1, E, pua);
        // solve_elastic_problem (flg2);

        solve_approx_cell_elastic_problem (flg1, E, pua, R, refine_cell, number_of_approx, f_name_cell);
        solve_elastic_problem (flg2, refine_hole, f_name_cell, f_name_hole);

        // cst num_cells = 10;
        cst num_of_rez_points = 10000;


        ATools::FourthOrderTensor C;
        arr<arr<vec<dbl>, 2>, 2> deform_line_1;
        arr<arr<arr<vec<dbl>, 2>, 2>, 2> deform_line_2;
        vec<dealii::Point<2>> coor_line_hole;
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_line_stress(3);
        OnCell::ArrayWithAccessToVector<arr<arr<arr<vec<dbl>, 3>, 3>, 3>> cell_line_deform(3);
        vec<dealii::Point<3>> coor_line_cell;


        get_line_deform_and_stress (
                ort_slice,
                coor_slice,
                C,
                deform_line_1,
                deform_line_2,
                coor_line_hole,
                cell_line_stress,
                cell_line_deform,
                coor_line_cell,
                number_of_approx,
                f_name_cell, f_name_hole
                );
        return;



        /////////////////////////////// собственно вычисление необходимых напряжений

        cst size_line_hole = coor_line_hole.size();
        cst size_line_cell = coor_line_cell.size();

        cdbl cell_size = 1.0 / num_cells;
        for (st i = 0; i < size_line_cell; ++i) //local coor cell to global
        {
            coor_line_cell[i](x) /= num_cells;
        };

        dbl max_macro_stress = 0.0;
        dbl max_final_stress = 0.0;

        arr<arr<vec<dbl>, 3>, 3> macro_stress;
        arr<arr<vec<dbl>, 3>, 3> final_stress;
        arr<arr<vec<dbl>, 3>, 3> final_deform;
        arr<arr<vec<dbl>, 3>, 3> final_stress_2;
        arr<arr<vec<dbl>, 3>, 3> final_deform_2;
        for (st i = 0; i < 3; ++i)
        {
           for (st j = 0; j < 3; ++j)
           {
               macro_stress[i][j] .resize (num_of_rez_points);
               final_stress[i][j] .resize (num_of_rez_points);
               final_deform[i][j] .resize (num_of_rez_points);
               final_stress_2[i][j] .resize (num_of_rez_points);
               final_deform_2[i][j] .resize (num_of_rez_points);
           }; 
        };
        {
            std::ofstream F(("hole_plas_cell_"+f_name_cell+"_"+f_name_hole+".gpd").c_str(),
                    std::ios::out);
            // FILE *F;
            // F = fopen("hole_plas_cell.gpd", "w");
            // F = fopen(("hole_plas_cell_"+f_name_cell+"_"+f_name_hole+".gpd").c_str(), "w");
            // for (st i = 0; i < size_line_hole; ++i)
            for (st i = 0; i < num_of_rez_points+1; ++i)
            {
                // dbl coor_in_cell = coor_line_hole[i](x);
                dbl coor_in_hole = 1.0 / num_of_rez_points * i;
                dbl coor_in_cell = coor_in_hole;
                for (st j = 0; j < num_cells; ++j)
                {
                    // printf("%f\n", coor_in_cell);
                    coor_in_cell -= cell_size;
                    if (coor_in_cell < 0.0)
                    {
                        coor_in_cell += cell_size;
                        break;
                    };
                };

                st point_number_in_cell = 0;
                for (st j = 0; j < size_line_cell; ++j)
                {
                    if (coor_in_cell < coor_line_cell[j](x))
                    {
                        point_number_in_cell = j;
                        break;
                    };
                };
                st point_number_in_hole = 0;
                for (st j = 0; j < size_line_hole; ++j)
                {
                    if ((coor_in_hole) < coor_line_hole[j](x))
                    {
                        point_number_in_hole = j;
                        break;
                    };
                };

                auto stress_in_cell  = [&coor_in_cell, &coor_line_cell, &cell_line_stress, point_number_in_cell] 
                    (cst i, cst j, cst k, cst l, cst m, cst n)
                    {
                        cst nm = point_number_in_cell;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_cell.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_cell;
                        cdbl X1 = coor_line_cell[nm_1](0);
                        cdbl X2 = coor_line_cell[nm](0);
                        cdbl Y1 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm_1];
                        cdbl Y2 = cell_line_stress[arr<i32,3>{i,j,k}][l][m][n][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                auto deform_in_cell  = [&coor_in_cell, &coor_line_cell, &cell_line_deform, point_number_in_cell] 
                    (cst i, cst j, cst k, cst l, cst m, cst n)
                    {
                        cst nm = point_number_in_cell;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_cell.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_cell;
                        cdbl X1 = coor_line_cell[nm_1](0);
                        cdbl X2 = coor_line_cell[nm](0);
                        cdbl Y1 = cell_line_deform[arr<i32,3>{i,j,k}][l][m][n][nm_1];
                        cdbl Y2 = cell_line_deform[arr<i32,3>{i,j,k}][l][m][n][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                        // return Y1;
                    };
                auto deform_in_hole  = [&coor_in_hole, &coor_line_hole, &deform_line_1, point_number_in_hole] 
                    (cst i, cst j)
                    {
                        cst nm = point_number_in_hole;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_hole.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_hole;
                        cdbl X1 = coor_line_hole[nm_1](0);
                        cdbl X2 = coor_line_hole[nm](0);
                        cdbl Y1 = deform_line_1[i][j][nm_1];
                        cdbl Y2 = deform_line_1[i][j][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                         // return Y1;
                    };
                auto deform_2_in_hole  = [&coor_in_hole, &coor_line_hole,
                     &deform_line_2, point_number_in_hole] 
                    (cst i, cst j, cst k)
                    {
                        cst nm = point_number_in_hole;
                        st nm_1 = 0;
                        if (nm == 0)
                            nm_1 = coor_line_hole.size() - 1;
                        else
                            nm_1 = nm - 1;
                        cdbl X = coor_in_hole;
                        cdbl X1 = coor_line_hole[nm_1](0);
                        cdbl X2 = coor_line_hole[nm](0);
                        cdbl Y1 = deform_line_2[i][j][k][nm_1];
                        cdbl Y2 = deform_line_2[i][j][k][nm];
                        return (X*(Y1-Y2)+X1*Y2-X2*Y1) / (X1-X2);
                         // return Y1;
                    };

                for (st alpha = 0; alpha < 3; ++alpha)
                {
                    for (st beta = 0; beta < 3; ++beta)
                    {
                       macro_stress[alpha][beta][i] = 
                           C[x][x][alpha][beta] * deform_in_hole(x,x) +
                           C[y][x][alpha][beta] * deform_in_hole(y,x) +
                           C[x][y][alpha][beta] * deform_in_hole(x,y) +
                           C[y][y][alpha][beta] * deform_in_hole(y,y);
                       final_stress[alpha][beta][i] = 
                           stress_in_cell(1,0,0,x,alpha,beta) * deform_in_hole(x,x) +
                           stress_in_cell(1,0,0,y,alpha,beta) * deform_in_hole(y,x) +
                           stress_in_cell(0,1,0,x,alpha,beta) * deform_in_hole(x,y) +
                           stress_in_cell(0,1,0,y,alpha,beta) * deform_in_hole(y,y);
                       final_deform[alpha][beta][i] = 
                           deform_in_cell(1,0,0,x,alpha,beta) * deform_in_hole(x,x) +
                           deform_in_cell(1,0,0,y,alpha,beta) * deform_in_hole(y,x) +
                           deform_in_cell(0,1,0,x,alpha,beta) * deform_in_hole(x,y) +
                           deform_in_cell(0,1,0,y,alpha,beta) * deform_in_hole(y,y);
                       final_stress_2[alpha][beta][i] = 
                           stress_in_cell(1,0,0,x,alpha,beta) * deform_in_hole(x,x) +
                           stress_in_cell(1,0,0,y,alpha,beta) * deform_in_hole(x,y) +
                           stress_in_cell(0,1,0,x,alpha,beta) * deform_in_hole(y,x) +
                           stress_in_cell(0,1,0,y,alpha,beta) * deform_in_hole(y,y) +
                           (
                           stress_in_cell(2,0,0,x,alpha,beta) * deform_2_in_hole(x,x,x) +
                           stress_in_cell(1,1,0,x,alpha,beta) * deform_2_in_hole(x,y,x) +
                           stress_in_cell(1,1,0,x,alpha,beta) * deform_2_in_hole(x,x,y) +
                           stress_in_cell(0,2,0,x,alpha,beta) * deform_2_in_hole(x,y,y) +
                           stress_in_cell(2,0,0,y,alpha,beta) * deform_2_in_hole(y,x,x) +
                           stress_in_cell(1,1,0,y,alpha,beta) * deform_2_in_hole(y,y,x) +
                           stress_in_cell(1,1,0,y,alpha,beta) * deform_2_in_hole(y,x,y) +
                           stress_in_cell(0,2,0,y,alpha,beta) * deform_2_in_hole(y,y,y)
                           ) * cell_size;
                       final_deform_2[alpha][beta][i] = 
                           deform_in_cell(1,0,0,x,alpha,beta) * deform_in_hole(x,x) +
                           deform_in_cell(1,0,0,y,alpha,beta) * deform_in_hole(x,y) +
                           deform_in_cell(0,1,0,x,alpha,beta) * deform_in_hole(y,x) +
                           deform_in_cell(0,1,0,y,alpha,beta) * deform_in_hole(y,y) +
                           (
                           deform_in_cell(2,0,0,x,alpha,beta) * deform_2_in_hole(x,x,x) +
                           deform_in_cell(1,1,0,x,alpha,beta) * deform_2_in_hole(x,y,x) +
                           deform_in_cell(1,1,0,x,alpha,beta) * deform_2_in_hole(x,x,y) +
                           deform_in_cell(0,2,0,x,alpha,beta) * deform_2_in_hole(x,y,y) +
                           deform_in_cell(2,0,0,y,alpha,beta) * deform_2_in_hole(y,x,x) +
                           deform_in_cell(1,1,0,y,alpha,beta) * deform_2_in_hole(y,y,x) +
                           deform_in_cell(1,1,0,y,alpha,beta) * deform_2_in_hole(y,x,y) +
                           deform_in_cell(0,2,0,y,alpha,beta) * deform_2_in_hole(y,y,y)
                           ) * cell_size;
                    };
                };
                arr<dbl, 44> out_data;
                out_data[0]  = coor_in_hole, 
                out_data[1]  = final_stress[x][x][i],
                out_data[2]  = final_stress[x][y][i],
                out_data[3]  = final_stress[y][y][i],
                out_data[4]  = stress_in_cell(1,0,0,x,x,x), 
                out_data[5]  = stress_in_cell(1,0,0,x,x,y), 
                out_data[6]  = stress_in_cell(1,0,0,x,y,y),
                out_data[7]  = stress_in_cell(0,1,0,x,x,x), 
                out_data[8]  = stress_in_cell(0,1,0,x,x,y), 
                out_data[9]  = stress_in_cell(0,1,0,x,y,y),
                out_data[10] = stress_in_cell(1,0,0,y,x,x), 
                out_data[11] = stress_in_cell(1,0,0,y,x,y), 
                out_data[12] = stress_in_cell(1,0,0,y,y,y),
                out_data[13] = stress_in_cell(0,1,0,y,x,x), 
                out_data[14] = stress_in_cell(0,1,0,y,x,y), 
                out_data[15] = stress_in_cell(0,1,0,y,y,y),
                out_data[16] = macro_stress[x][x][i],
                out_data[17] = macro_stress[x][y][i],
                out_data[18] = macro_stress[y][x][i],
                out_data[19] = macro_stress[y][y][i],
                out_data[20] = final_deform[x][x][i],
                out_data[21] = final_deform[x][y][i],
                out_data[22] = final_deform[y][y][i],
                out_data[23] = deform_in_cell(1,0,0,x,x,x), 
                out_data[24] = deform_in_cell(1,0,0,x,x,y), 
                out_data[25] = deform_in_cell(1,0,0,x,y,y),
                out_data[26] = deform_in_cell(0,1,0,x,x,x), 
                out_data[27] = deform_in_cell(0,1,0,x,x,y), 
                out_data[28] = deform_in_cell(0,1,0,x,y,y),
                out_data[29] = deform_in_cell(1,0,0,y,x,x), 
                out_data[30] = deform_in_cell(1,0,0,y,x,y), 
                out_data[31] = deform_in_cell(1,0,0,y,y,y),
                out_data[32] = deform_in_cell(0,1,0,y,x,x), 
                out_data[33] = deform_in_cell(0,1,0,y,x,y), 
                out_data[34] = deform_in_cell(0,1,0,y,y,y);
                out_data[35] = final_stress_2[x][x][i],
                out_data[36] = final_stress_2[x][y][i],
                out_data[37] = final_stress_2[y][y][i],
                out_data[38] = final_deform_2[x][x][i],
                out_data[39] = final_deform_2[x][y][i],
                out_data[40] = final_deform_2[y][y][i],
                out_data[41] = deform_in_hole(x,x),
                out_data[42] = deform_in_hole(x,y),
                out_data[43] = deform_in_hole(y,y);
                // fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                //         coor_in_hole, 
                //         final_stress[x][x][i],
                //         final_stress[x][y][i],
                //         final_stress[y][y][i],
                //         stress_in_cell(1,0,0,x,x,x), 
                //         stress_in_cell(1,0,0,x,x,y), 
                //         stress_in_cell(1,0,0,x,y,y),
                //         stress_in_cell(0,1,0,x,x,x), 
                //         stress_in_cell(0,1,0,x,x,y), 
                //         stress_in_cell(0,1,0,x,y,y),
                //         stress_in_cell(1,0,0,y,x,x), 
                //         stress_in_cell(1,0,0,y,x,y), 
                //         stress_in_cell(1,0,0,y,y,y),
                //         stress_in_cell(0,1,0,y,x,x), 
                //         stress_in_cell(0,1,0,y,x,y), 
                //         stress_in_cell(0,1,0,y,y,y),
                //         macro_stress[x][x][i],
                //         macro_stress[x][y][i],
                //         macro_stress[y][x][i],
                //         macro_stress[y][y][i],
                //         final_deform[x][x][i],
                //         final_deform[x][y][i],
                //         final_deform[y][y][i],
                //         deform_in_cell(1,0,0,x,x,x), 
                //         deform_in_cell(1,0,0,x,x,y), 
                //         deform_in_cell(1,0,0,x,y,y),
                //         deform_in_cell(0,1,0,x,x,x), 
                //         deform_in_cell(0,1,0,x,x,y), 
                //         deform_in_cell(0,1,0,x,y,y),
                //         deform_in_cell(1,0,0,y,x,x), 
                //         deform_in_cell(1,0,0,y,x,y), 
                //         deform_in_cell(1,0,0,y,y,y),
                //         deform_in_cell(0,1,0,y,x,x), 
                //         deform_in_cell(0,1,0,y,x,y), 
                //         deform_in_cell(0,1,0,y,y,y)
                //         );
                F 
                    << out_data[0] << " "
                    << out_data[1] << " "
                    << out_data[2] << " "
                    << out_data[3] << " "
                    << out_data[4] << " "
                    << out_data[5] << " "
                    << out_data[6] << " "
                    << out_data[7] << " "
                    << out_data[8] << " "
                    << out_data[9] << " "
                    << out_data[10] << " "
                    << out_data[11] << " "
                    << out_data[12] << " "
                    << out_data[13] << " "
                    << out_data[14] << " "
                    << out_data[15] << " "
                    << out_data[16] << " "
                    << out_data[17] << " "
                    << out_data[18] << " "
                    << out_data[19] << " "
                    << out_data[20] << " "
                    << out_data[21] << " "
                    << out_data[22] << " "
                    << out_data[23] << " "
                    << out_data[24] << " "
                    << out_data[25] << " "
                    << out_data[26] << " "
                    << out_data[27] << " "
                    << out_data[28] << " "
                    << out_data[29] << " "
                    << out_data[30] << " "
                    << out_data[31] << " "
                    << out_data[32] << " "
                    << out_data[33] << " "
                    << out_data[34] << " "
                    << out_data[35] << " "
                    << out_data[36] << " "
                    << out_data[37] << " "
                    << out_data[38] << " "
                    << out_data[39] << " "
                    << out_data[40] << " "
                    << out_data[41] << " "
                    << out_data[42] << " "
                    << out_data[43] << " "
                    << std::endl;
                // for (st i = 0; i < num_of_rez_points; ++i)
                // {
                //     if ((coor_in_hole > cell_size * (43-1)) and (coor_in_hole < cell_size * (43)))
                //     // if ((coor_in_hole > cell_size * (22-1)) and (coor_in_hole < cell_size * (22)))
                //             {
                //     if (max_macro_stress < macro_stress[y][y][i])
                //         max_macro_stress = macro_stress[y][y][i];
                //     if (max_final_stress < final_stress[y][y][i])
                //         max_final_stress = final_stress[y][y][i];
                //         };
                // };
            };
            // fclose(F);
            F.close ();
        };
        // {
        //     FILE *f;
        //     f = fopen("pua_25_2.gpd", "a");
        //     fprintf(f, "%f %f %f\n", pua, max_macro_stress, max_final_stress);
        //     fclose(f);
        // };
    };
};

int main()
{
    //heat_conduction_problem
    solve_heat_conduction_problem (0); //!!!!!!!!!!!!!!!!!!!!!!!!!1

    //heat_conduction_problem_on_cell
    solve_heat_conduction_problem_on_cell (0);

    //elasstic_problem
    solve_elastic_problem (0);

    // elasstic_problem_on_cell
    solve_elastic_problem_on_cell (0);
    
    //heat_conduction_nikola_problem
    solve_heat_conduction_nikola_problem (0);

    //nikola_elasstic_problem
    solve_nikola_elastic_problem (0);
    

    //heat_conduction_problem_3d
    solve_heat_conduction_problem_3d (0);

    //heat_conduction_problem_on_cell_3d
    solve_heat_conduction_problem_on_cell_3d (1);

    // elasstic_problem_3d
    solve_elastic_problem_3d (0);

    // elasstic_problem_on_cell_3d
    solve_elastic_problem_on_cell_3d (0);

    // solve_heat_conduction_problem_on_cell (1);
    solve_approx_cell_heat_problem (0);
    printf(" \n");

    solve_approx_cell_elastic_problem (0);

    solve_two_stress (0);

    solve_cell_elastic_problem_and_print_along_line(0);


    {
        cst n_ref = 5;
        cst n_ref_real = 7;

        cdbl ratio = 8.0;
        cdbl H = 1.0;
        cdbl W = 5.0;
        cdbl Ri = 20.0;
        cdbl Ro = Ri + W;
        cdbl P = 5.0;//1.0 / 8.0;// / ratio;
        cdbl R_fiber = 0.25;
        cst n_p = 64;

        cst Ncx = 5;
        cst Ncy = 5; //Ncx * ratio;
        cdbl bx = W / Ncx;
        cdbl by = 1.0 / Ncy;
        cst Npx = 100;
        cst Npy = 100;
        cdbl dx = W / (Npx-1);
        cdbl dy = 1.0 / (Npy-1);
        solve_approx_cell_elastic_problem (0, 10.0, 0.25, R_fiber, n_p);
        solve_ring_problem_3d<n_ref>(0, H, W, Ri, 1 << 4, P);
        calculate_real_stress_in_ring_arbitrary_grid_alternate<n_ref, n_ref_real>(
                0, H, W, Ri, Ncx, Ncy, R_fiber, n_p);
    };


    solve_plate_with_hole_problem (cst flag)
    
    // vec<vec<dbl>> A(3);
    // for (st i = 0; i < 3; ++i)
    // {
    //     A[i] .resize(3);
    // };
    // vec<dbl> X(3);
    // vec<dbl> b(3);
    //
    // X[0] = 4.0; X[1] = 6.0; X[2] = 3.0;
    // for (st i = 0; i < 3; ++i)
    // {
    //     for (st j = 0; j < 3; ++j)
    //     {
    //         A[i][j] = i*j*1.0 + 1.0;
    //     };
    // };
    //
    // for (st i = 0; i < 3; ++i)
    // {
    //     b[i] = 
    //         A[i][0] * X[0] +
    //         A[i][1] * X[1] +
    //         A[i][2] * X[2];
    // };
    //
    // gaus_solve (A, X, b);
    //
    // std::cout << X[0] << " " << X[1] << " " << X[2] << std::endl;
    

    // solve_approx_cell_elastic_problem (1, 10.0, 0.25);

    // calculate_real_stress (1, 50, 2.0, 0.25, 0, 1, 0.125, 4, 8,
    //         str("E_2_R_125_4"), str("E_2_R_125_4_8"));
    //
    // calculate_real_stress (1, 50, 10.0, 0.25, 0, 1, 0.125, 4, 8,
    //         str("E_10_R_125_4"), str("E_10_R_125_4_8"));
    //
    // calculate_real_stress (1, 50, 50.0, 0.25, 0, 1, 0.125, 4, 8,
    //         str("E_50_R_125_4"), str("E_50_R_125_4_8"));


    // calculate_real_stress (1, 100, 2.0, 0.25, 0, 1, 0.25, 4, 8,
    //         str("E_2_R_25_4"), str("E_2_R_25_4_8"));
    //
    // calculate_real_stress (1, 100, 10.0, 0.25, 1, 1, 0.25, 5, 9,
    //         str("E_10_R_25_5"), str("E_10_R_25_4_9"));
    //
    // calculate_real_stress (1, 100, 50.0, 0.25, 0, 1, 0.25, 4, 8,
    //         str("E_50_R_25_4"), str("E_50_R_25_4_8"));


    // calculate_real_stress (1, 150, 2.0, 0.25, 0, 1, 0.375, 4, 8,
    //         str("E_2_R_375_4"), str("E_2_R_375_4_8"));
    //
    // calculate_real_stress (1, 150, 10.0, 0.25, 0, 1, 0.375, 4, 8,
    //         str("E_10_R_375_4"), str("E_10_R_375_4_8"));
    //
    // calculate_real_stress (1, 150, 50.0, 0.25, 0, 1, 0.375, 4, 8,
    //         str("E_50_R_375_4"), str("E_50_R_375_4_8"));

    // calculate_real_stress (1, 100, 10.0, 0.25, 0, 1, 0.25, 5, 7,
    //         str("E_10_R_25_5"), str("E_10_R_25_5_7_without_hole"));

    // calculate_real_stress (1, 100, 10.0, 0.25, 0, 0, 0.25, 5, 9,
    //         str("E_10_R_25_5"), str("E_10_R_25_5_9"));

    // calculate_real_stress (1, 100, 10.0, 0.25, 1, 1, 0.25, 4, 7,
    //         str("E_10_R_25_4"), str("E_10_R_25_4_7"));

    // calculate_real_stress (1, 100, 10.0, 0.25, 1, 0, 0.25, 4, 8,
    //         str("E_10_R_25_4_ortotrop"), str("E_10_R_25_4_8"));

    // calculate_real_stress (1, 100, 10.0, 0.25, 1, 0, 0.25, 4, 8,
    //         str("E_10_R_25_4"), str("E_10_R_25_4_8"));
    
    enum {x, y, z};//_ortotrop


    // if (!access("lol", 0))
    // {
    //     printf("AGA\n");
    // }
    // else
    // {
    //     printf("NEA\n");
    //     mkdir("lol", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    // };


    // solve_approx_cell_elastic_problem (1, 2.0, 0.25, 0.125, "not_isotrop_125.gpd");
    // solve_elastic_problem (1);
    // solve_two_stress (1, 2.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 5.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 5.0, 0.25);
    //
    solve_approx_cell_elastic_problem (0, 10.0, 0.25);
    // solve_approx_cell_elastic_problem (1);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 20.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 20.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 50.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 100.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 100.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 100.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 200.0, 0.25);
    
    // solve_approx_cell_elastic_problem (1, 10.0, 0.1);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.1);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.15);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.15);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.20);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.20);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.3);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 10.0, 0.30);
    
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.1);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.1);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.15);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.15);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.20);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.20);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.25);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.25);
    //
    // solve_approx_cell_elastic_problem (1, 10.0, 0.3);
    // solve_elastic_problem (1);
    // solve_two_stress (1, 50.0, 0.30);

    // solve_approx_cell_elastic_problem (1, 2.0, 0.25, 0.125, "not_isotrop_125.gpd");
    // solve_approx_cell_elastic_problem (1, 5.0, 0.25, 0.125, "not_isotrop_125.gpd");
    // solve_approx_cell_elastic_problem (1, 10.0, 0.25, 0.125, "not_isotrop_125.gpd");
    // solve_approx_cell_elastic_problem (1, 20.0, 0.25, 0.125, "not_isotrop_125.gpd");
    // solve_approx_cell_elastic_problem (1, 100.0, 0.25, 0.125, "not_isotrop_125.gpd");
    //
    // solve_approx_cell_elastic_problem (1, 2.0, 0.25, 0.25, "not_isotrop_25.gpd");
    // solve_approx_cell_elastic_problem (1, 5.0, 0.25, 0.25, "not_isotrop_25.gpd");
    // solve_approx_cell_elastic_problem (1, 10.0, 0.25, 0.25, "not_isotrop_25.gpd");
    // solve_approx_cell_elastic_problem (1, 20.0, 0.25, 0.25, "not_isotrop_25.gpd");
    // solve_approx_cell_elastic_problem (1, 100.0, 0.25, 0.25, "not_isotrop_25.gpd");
    //
    // solve_approx_cell_elastic_problem (1, 2.0, 0.25, 0.375, "not_isotrop_375.gpd");
    // solve_approx_cell_elastic_problem (1, 5.0, 0.25, 0.375, "not_isotrop_375.gpd");
    // solve_approx_cell_elastic_problem (1, 10.0, 0.25, 0.375, "not_isotrop_375.gpd");
    // solve_approx_cell_elastic_problem (1, 20.0, 0.25, 0.375, "not_isotrop_375.gpd");
    // solve_approx_cell_elastic_problem (1, 100.0, 0.25, 0.375, "not_isotrop_375.gpd");
    // solve_approx_cell_elastic_problem (1, 2.0, 0.25, 0.25, "not_isotrop_25.gpd");

    // if (0)
    // {
    //     {
    //         // atools::fourthordertensor e;
    //         // atools::fourthordertensor c;
    //         // for (st i = 0; i < 3; ++i)
    //         // {
    //         //     for (st j = 0; j < 3; ++j)
    //         //     {
    //         //         for (st k = 0; k < 3; ++k)
    //         //         {
    //         //             for (st l = 0; l < 3; ++l)
    //         //             {
    //         //                 e[i][j][k][l] = 0.0;
    //         //                 c[i][j][k][l] = 0.0;
    //         //             };
    //         //         };
    //         //     }; 
    //         // };
    //         arr<arr<dbl, 6>, 6> e2d_original;
    //         // arr<arr<dbl, 6>, 6> e2d_final;
    //         for (st i = 0; i < 6; ++i)
    //         {
    //             for (st j = 0; j < 6; ++j)
    //             {
    //                 e2d_original[i][j] = 0.0;
    //             };
    //         };
    //
    //         arr<dbl, 3> e = {10.0, 10.0, 100.0};
    //
    //         e2d_original[0][0] = 1.0 / e[0];
    //         e2d_original[1][1] = 1.0 / e[1];
    //         e2d_original[2][2] = 1.0 / e[2];
    //         e2d_original[0][1] = -0.25 / e[0];
    //         e2d_original[0][2] = -0.25 / e[0];
    //         e2d_original[1][0] = -0.25 / e[1];
    //         e2d_original[1][2] = -0.25 / e[1];
    //         e2d_original[2][0] = -0.25 / e[2];
    //         e2d_original[2][1] = -0.25 / e[2];
    //         e2d_original[3][3] = 0.5 / 4.0;
    //         e2d_original[4][4] = 0.5 / 4.0;
    //         e2d_original[5][5] = 0.5 / 4.0;
    //
    //         auto c2d_original = inverse (e2d_original);
    //         auto c = t2_to_t4 (c2d_original);
    //         auto c2d_final = t4_to_t2 (c);
    //         auto e2d_final = inverse (c2d_final);
    //         auto newcoef = unphysical_to_physicaly (c);
    //
    //         printf("e2d_original\n");
    //         for (size_t i = 0; i < 6; ++i)
    //         {
    //             for (size_t j = 0; j < 6; ++j)
    //             {
    //                 if (std::abs(e2d_original[i][j]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             e2d_original[i][j]);
    //                 else
    //                     printf("%f   ", 
    //                             e2d_original[i][j]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //
    //         printf("c2d_original\n");
    //         for (size_t i = 0; i < 6; ++i)
    //         {
    //             for (size_t j = 0; j < 6; ++j)
    //             {
    //                 if (std::abs(c2d_original[i][j]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             c2d_original[i][j]);
    //                 else
    //                     printf("%f   ", 
    //                             c2d_original[i][j]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //
    //         printf("c\n");
    //         for (size_t i = 0; i < 9; ++i)
    //         {
    //             uint8_t im = i / (2 + 1);
    //             uint8_t in = i % (2 + 1);
    //
    //             for (size_t j = 0; j < 9; ++j)
    //             {
    //                 uint8_t jm = j / (2 + 1);
    //                 uint8_t jn = j % (2 + 1);
    //
    //                 if (std::abs(c[im][in][jm][jn]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             c[im][in][jm][jn]);
    //                 else
    //                     printf("%f   ", 
    //                             c[im][in][jm][jn]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //
    //         printf("e2d_final\n");
    //         for (size_t i = 0; i < 6; ++i)
    //         {
    //             for (size_t j = 0; j < 6; ++j)
    //             {
    //                 if (std::abs(e2d_final[i][j]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             e2d_final[i][j]);
    //                 else
    //                     printf("%f   ", 
    //                             e2d_final[i][j]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //
    //         printf("c2d_final\n");
    //         for (size_t i = 0; i < 6; ++i)
    //         {
    //             for (size_t j = 0; j < 6; ++j)
    //             {
    //                 if (std::abs(c2d_final[i][j]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             c2d_final[i][j]);
    //                 else
    //                     printf("%f   ", 
    //                             c2d_final[i][j]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //
    //         printf("newcoef\n");
    //         for (size_t i = 0; i < 9; ++i)
    //         {
    //             uint8_t im = i / (2 + 1);
    //             uint8_t in = i % (2 + 1);
    //
    //             for (size_t j = 0; j < 9; ++j)
    //             {
    //                 uint8_t jm = j / (2 + 1);
    //                 uint8_t jn = j % (2 + 1);
    //
    //                 if (std::abs(newcoef[im][in][jm][jn]) > 0.0000001)
    //                     printf("\x1b[31m%f\x1b[0m   ", 
    //                             newcoef[im][in][jm][jn]);
    //                 else
    //                     printf("%f   ", 
    //                             newcoef[im][in][jm][jn]);
    //             };
    //             for (size_t i = 0; i < 2; ++i)
    //                 printf("\n");
    //         };
    //         printf("\n");
    //     auto ea = t2_to_t4 (e2d_final);
    //     printf("%f %f %f %f %f %f %f %f %f\n", 
    //             1.0/ea[0][0][0][0],
    //             -ea[0][0][1][1]/ea[0][0][0][0],
    //             -ea[0][0][2][2]/ea[0][0][0][0],
    //             -ea[1][1][0][0]/ea[1][1][1][1],
    //             1.0/ea[1][1][1][1],
    //             -ea[1][1][2][2]/ea[1][1][1][1],
    //             -ea[2][2][0][0]/ea[2][2][2][2],
    //             -ea[2][2][1][1]/ea[2][2][2][2],
    //             1.0/ea[2][2][2][2]
    //             );
    //     printf("%f %f %f %f %f %f %f %f %f\n", 
    //             newcoef[0][0][0][0],
    //             newcoef[0][0][1][1],
    //             newcoef[0][0][2][2],
    //             newcoef[1][1][0][0],
    //             newcoef[1][1][1][1],
    //             newcoef[1][1][2][2],
    //             newcoef[2][2][0][0],
    //             newcoef[2][2][1][1],
    //             newcoef[2][2][2][2]
    //             );
    //     };
    // };

    // arr<arr<dbl,3>,3> e = {arr<dbl,3>{1.0/100.0, -0.25/100.0, -0.25/100.0},
    //                        arr<dbl,3>{-0.25/10.0, 1.0/10.0, -0.25/10.0},
    //                        arr<dbl,3>{-0.25/10.0, -0.25/10.0, 1.0/10.0}};
    // auto ie = inverse(e);

    // arr<arr<dbl,3>,3> i;
    // for_i(0, 3)
    //     for_j(0, 3)
    //     i[i][j] = 0.0;
    // for_i(0, 3)
    //     for_j(0, 3)
    //     {
    //         double temp = 0.0;
    //         for_k(0, 3)
    //             temp += e[i][k] * ie[k][j];
    //         i[i][j] = temp;
    //     };
    //
    // printf("\n");
    //
    // printf("%f %f %f\n", e[0][0], e[0][1], e[0][2]);
    // printf("%f %f %f\n", e[1][0], e[1][1], e[1][2]);
    // printf("%f %f %f\n", e[2][0], e[2][1], e[2][2]);
    //
    // printf("\n");
    //
    // printf("%f %f %f\n", ie[0][0], ie[0][1], ie[0][2]);
    // printf("%f %f %f\n", ie[1][0], ie[1][1], ie[1][2]);
    // printf("%f %f %f\n", ie[2][0], ie[2][1], ie[2][2]);
    //
    // printf("\n");
    //
    // printf("%f %f %f\n", i[0][0], i[0][1], i[0][2]);
    // printf("%f %f %f\n", i[1][0], i[1][1], i[1][2]);
    // printf("%f %f %f\n", i[2][0], i[2][1], i[2][2]);
    // arr<arr<dbl,3>,3> e = {arr<dbl,3>{10.0, 0.30, 0.15},
    //                        arr<dbl,3>{0.30, 10.0, 0.15},
    //                        arr<dbl,3>{0.15, 0.15, 20.0}};
    // arr<dbl,3> g = {4.0, 4.0, 4.0};
    // atools::fourthordertensor e;
    // for (st i = 0; i < 3; ++i)
    // {
    //    for (st j = 0; j < 3; ++j)
    //    {
    //        for (st k = 0; k < 3; ++k)
    //        {
    //            for (st l = 0; l < 3; ++l)
    //            {
    //                e[i][j][k][l] = 0.0;
    //            };
    //        };
    //    }; 
    // };
    // e[x][x][x][x] = 1 /
    // atools::fourthordertensor c;
    // for (st i = 0; i < 3; ++i)
    // {
    //    for (st j = 0; j < 3; ++j)
    //    {
    //        for (st k = 0; k < 3; ++k)
    //        {
    //            for (st l = 0; l < 3; ++l)
    //            {
    //                c[i][j][k][l] = 0.0;
    //            };
    //        };
    //    }; 
    // };
    // c[x][x][x][x] = ie[x][x];
    // c[x][x][y][y] = ie[x][y];
    // c[x][x][z][z] = ie[x][z];
    // c[y][y][x][x] = ie[y][x];
    // c[y][y][y][y] = ie[y][y];
    // c[y][y][z][z] = ie[y][z];
    // c[z][z][x][x] = ie[z][x];
    // c[z][z][y][y] = ie[z][y];
    // c[z][z][z][z] = ie[z][z];
    //
    // c[x][y][x][y] = 4.0;
    // c[y][x][y][x] = 4.0;
    // c[x][z][x][z] = 4.0;
    // c[z][x][z][x] = 4.0;
    // c[y][z][y][z] = 4.0;
    // c[z][y][z][y] = 4.0;
    // // eptools::set_ortotropic_elascity(e, g, c);
    //     // eptools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(c);
    //     for (size_t i = 0; i < 9; ++i)
    //     {
    //         uint8_t im = i / (2 + 1);
    //         uint8_t in = i % (2 + 1);
    //
    //         for (size_t j = 0; j < 9; ++j)
    //         {
    //             uint8_t jm = j / (2 + 1);
    //             uint8_t jn = j % (2 + 1);
    //
    //             if (std::abs(c[im][in][jm][jn]) > 0.0000001)
    //                 printf("\x1b[31m%f\x1b[0m   ", 
    //                         c[im][in][jm][jn]);
    //             else
    //                 printf("%f   ", 
    //                         c[im][in][jm][jn]);
    //         };
    //         for (size_t i = 0; i < 2; ++i)
    //             printf("\n");
    //     };
    // };
    //
    // printf("\n");
    // printf("\n");

    // {
    // arr<arr<dbl,3>,3> e = {arr<dbl,3>{100.0, 0.25, 0.25},
    //                        arr<dbl,3>{0.25, 10.0, 0.25},
    //                        arr<dbl,3>{0.25, 0.25, 10.0}};
    // arr<dbl,3> g = {4.0, 4.0, 4.0};
    // atools::fourthordertensor c;
    //     eptools::set_ortotropic_elascity(e, g, c);
    //     // eptools ::set_isotropic_elascity{yung : 10.0, puasson : 0.25}(c);
    //     for (size_t i = 0; i < 9; ++i)
    //     {
    //         uint8_t im = i / (2 + 1);
    //         uint8_t in = i % (2 + 1);
    //
    //         for (size_t j = 0; j < 9; ++j)
    //         {
    //             uint8_t jm = j / (2 + 1);
    //             uint8_t jn = j % (2 + 1);
    //
    //             if (std::abs(c[im][in][jm][jn]) > 0.0000001)
    //                 printf("\x1b[31m%f\x1b[0m   ", 
    //                         c[im][in][jm][jn]);
    //             else
    //                 printf("%f   ", 
    //                         c[im][in][jm][jn]);
    //         };
    //         for (size_t i = 0; i < 2; ++i)
    //             printf("\n");
    //     };
    // };
    //
    //
    //             printf("\n");
    // auto a = ::unphysical_to_physicaly(c);
    //     for (size_t i = 0; i < 9; ++i)
    //     {
    //         uint8_t im = i / (2 + 1);
    //         uint8_t in = i % (2 + 1);
    //
    //         for (size_t j = 0; j < 9; ++j)
    //         {
    //             uint8_t jm = j / (2 + 1);
    //             uint8_t jn = j % (2 + 1);
    //
    //             if (std::abs(a[im][in][jm][jn]) > 0.0000001)
    //                 printf("\x1b[31m%f\x1b[0m   ", 
    //                         a[im][in][jm][jn]);
    //             else
    //                 printf("%f   ", 
    //                         a[im][in][jm][jn]);
    //         };
    //         for (size_t i = 0; i < 2; ++i)
    //             printf("\n");
    //     };




    // st size = 0;
    // {
    //     std::ifstream in ("solution_size.bin", std::ios::in | std::ios::binary);
    //     in.read ((char *) &size, sizeof size);
    //     in.close ();
    // };
    // dealii::vector<dbl> sol(size);
    // {
    //     std::ifstream in ("solution_0.bin", std::ios::in | std::ios::binary);
    //     for (st i = 0; i < size; ++i)
    //     {
    //         in.read ((char *) &sol[i], sizeof(dbl));
    //     };
    //     in.close ();
    // };
    // printf("size %d %f\n", size, sol[10]);

    //
    // {
    //     arr<prmt::point<2>, 4> points = {
    //         // prmt::point<2>(0.0, 0.0),
    //         // prmt::point<2>(0.5, 0.5),
    //         // prmt::point<2>(0.0, 1.0),
    //         // prmt::point<2>(-0.5, 0.5)};
    //         prmt::point<2>(-0.5, -0.5),
    //         prmt::point<2>(0.5, -0.5),
    //         prmt::point<2>(1.0, 1.0),
    //         prmt::point<2>(0.0, 1.0)};
    //     arr<dbl, 4> values = {
    //         0.0, 1.0, 0.0, 1.0};
    //
    //     scalar4pointsfunc<2> f(points, values);
    //
    //     file *f;
    //
    //     f = fopen("test_4_point_value.gpd", "w");
    //     fprintf(f, "%lf %lf %lf\n", points[0].x(), points[0].y(), f(points[0]));
    //     fprintf(f, "%lf %lf %lf\n", points[1].x(), points[1].y(), f(points[1]));
    //     fprintf(f, "%lf %lf %lf\n", points[2].x(), points[2].y(), f(points[2]));
    //     fprintf(f, "%lf %lf %lf\n", points[3].x(), points[3].y(), f(points[3]));
    //     fclose(f);
    //
    //     f = fopen("test_value.gpd", "w");
    //     for (st i = 0; i < 100; ++i)
    //         for (st j = 0; j < 100; ++j)
    //         {
    //             fprintf(f, "%lf %lf %lf\n", -1.0+0.03*i, -1.0+0.03*j, f(-1.0+0.03*i,-1.0+0.03*j));
    //         };
    //     fclose(f);
    //
    //     f = fopen("test_4_point_grad.gpd", "w");
    //     fprintf(f, "%lf %lf %lf\n", points[0].x(), points[0].y(), f.dx(points[0]));
    //     fprintf(f, "%lf %lf %lf\n", points[1].x(), points[1].y(), f.dx(points[1]));
    //     fprintf(f, "%lf %lf %lf\n", points[2].x(), points[2].y(), f.dx(points[2]));
    //     fprintf(f, "%lf %lf %lf\n", points[3].x(), points[3].y(), f.dx(points[3]));
    //     fclose(f);
    //
    //     f = fopen("test_grad.gpd", "w");
    //     for (st i = 0; i < 100; ++i)
    //         for (st j = 0; j < 100; ++j)
    //         {
    //             fprintf(f, "%lf %lf %lf\n", -1.0+0.03*i, -1.0+0.03*j, f.dx(-1.0+0.03*i,-1.0+0.03*j));
    //         };
    //     fclose(f);
    // };


    // debputs();
    // lmbd<st(cst)> add_i = [](cst i){return i-1;};
    // printf("%ld\n", foo(10, [](cst i){return i-1;}, [](cst i){return i+3;}));
    // arr<dealii::Point<2>, 4> quad;
    // quad[0] = dealii::Point<2>(0.0, 0.0);
    // quad[1] = dealii::Point<2>(1.0, 0.0);
    // quad[2] = dealii::Point<2>(1.0, 1.0);
    // quad[3] = dealii::Point<2>(0.0, 1.0);
    // dealii::Point<2> p = dealii::Point<2>(1.5, 0.5);
    // printf("in quad %d\n", point_in_quadrilateral(p, quad));
   //  st aa = 1;
   //  st bb = ++aa;
   //  printf("%ld %ld %ld %ld\n", 2_pow(0), aa++, ++aa, bb);
   //  {
   //  cubeofnumbers<2, 2> cube(5);
   //  printf("%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n%ld %ld %ld %ld %ld \n", 
   //          cube.number[0][4][0], cube.number[1][4][0], cube.number[2][4][0], cube.number[3][4][0], cube.number[4][4][0],
   //          cube.number[0][3][0], cube.number[1][3][0], cube.number[2][3][0], cube.number[3][3][0], cube.number[4][3][0],
   //          cube.number[0][2][0], cube.number[1][2][0], cube.number[2][2][0], cube.number[3][2][0], cube.number[4][2][0],
   //          cube.number[0][1][0], cube.number[1][1][0], cube.number[2][1][0], cube.number[3][1][0], cube.number[4][1][0],
   //          cube.number[0][0][0], cube.number[1][0][0], cube.number[2][0][0], cube.number[3][0][0], cube.number[4][0][0]);
   //  };
   //  printf("\n");
   //  {
   //  cubeofnumbers<3, 3> cube(1);
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
    // {
    //     Domain<3> domain;
    //     {
    //         dealii::gridgenerator::hyper_cube(domain.grid, 0.0, 2.0);
    //         domain.grid.refine_global(1);
    //     };
    // printf("level %d\n", domain.grid.n_global_levels());
    // };

    // {
    //     dealii::triangulation<2> tria;
    //     vec<prmt::point<2>> outer(4);//3,0000000000000004440892098500626161694527
    //     vec<prmt::point<2>> inner(4);

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
    //         dealii::gridout grid_out;
    //         grid_out.write_eps (tria, out);
    //     };
    // };

    // printf("true? %d\n", 9.0000000000000006 == 9.0000000000000008);
    // puts("true? 0.000000000000000");
    // printf("true? %.50f\n", 3.0);
    // printf("true? %.50f\n", 3.000000000000007);
    // printf("true? %.50f\n", 3.0000000000000007);
    // printf("true? %.50f\n", 3.0000000000000007);
    // printf("true? %.50f\n", (3.0000000000000007 - 3.0));




    // {
    //     vec<atools::fourthordertensor> c(2);
    //     eptools ::set_isotropic_elascity{yung : 1.0, puasson : 0.34}(c[0]);
    //     eptools ::set_isotropic_elascity{yung : 5.0, puasson : 0.30}(c[1]);
    //     
    //     auto arith = [c] (cdbl area) {
    //         atools::fourthordertensor res;
    //         for (st i = 0; i < 3; ++i)
    //         {
    //             for (st j = 0; j < 3; ++j)
    //             {
    //                 for (st k = 0; k < 3; ++k)
    //                 {
    //                     for (st l = 0; l < 3; ++l)
    //                     {
    //                         res[i][j][k][l] = 
    //                             c[0][i][j][k][l] * (1.0 - area) + 
    //                             c[1][i][j][k][l] * area;
    //                     };
    //                 };
    //             };
    //         };
    //         return res;
    //     };

    //     auto harm = [c] (cdbl area) {
    //         atools::fourthordertensor res;
    //         for (st i = 0; i < 3; ++i)
    //         {
    //             for (st j = 0; j < 3; ++j)
    //             {
    //                 for (st k = 0; k < 3; ++k)
    //                 {
    //                     for (st l = 0; l < 3; ++l)
    //                     {
    //                         res[i][j][k][l] = 1.0 /
    //                             (c[0][i][j][k][l] * area + 
    //                              c[1][i][j][k][l] * (1.0 - area));
    //                     };
    //                 };
    //             };
    //         };
    //         return res;
    //     };

    //     file *f1;
    //     file *f2;
    //     f1 = fopen("arith.gpd", "a");
    //     f2 = fopen("harm.gpd", "a");
    //     dbl size = 0.01;
    //     while (size < 0.999)
    //     {
    //         auto mean1 = arith(size*size);
    //         auto mean2 = harm(size*size);
    //         auto newcoef1 = unphysical_to_physicaly (mean1);
    //         auto newcoef2 = unphysical_to_physicaly (mean2);

    //         fprintf(f1, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
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

    //         fprintf(f2, "%f %f %f %f %f %f %f %f %f %f %f %f\n", size*size, 
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
    //     fclose(f1);
    //     fclose(f2);
    // };

    
    
//     {
//     arr<prmt::point<2>, 4> points1 = {
//         prmt::point<2>(0.0, 0.0),
//         prmt::point<2>(2.0, 0.0),
//         prmt::point<2>(1.5, 1.5),
//         prmt::point<2>(0.0, 2.0)};
//     arr<dbl, 4> values1 = {
//         points1[0].x()*points1[0].x(),
//         points1[1].x()*points1[1].x(),
//         points1[2].x()*points1[2].x(),
//         points1[3].x()*points1[3].x()};
// 
//     arr<prmt::point<2>, 4> points2 = {
//         prmt::point<2>(2.0, 0.0),
//         prmt::point<2>(4.0, 3.0),
//         prmt::point<2>(3.0, 4.0),
//         prmt::point<2>(1.5, 1.5)};
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
//     scalar4pointsfunc<2> f1(points1, values1);
// 
//     // points[0] = prmt::point<2>(1.0, 0.0);
//     // points[1] = prmt::point<2>(2.0, 0.0);
//     // points[2] = prmt::point<2>(3.0, 1.0);
//     // points[3] = prmt::point<2>(2.0, 1.0);
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
//     scalar4pointsfunc<2> f2(points2, values2);
// 
//     auto ip = prmt::point<2>(1.5, 1.5);
// 
//     // printf("%lf\n", f1(0.0, 0.0));
//     // printf("%lf\n", f1(ip));
//     // printf("%lf\n", f1.dx(ip));
//     // printf("%lf\n", f2(ip));
//     // printf("%lf\n", f2.dx(ip));
// 
// 
//     arr<arr<prmt::point<2>, 4>, 4> octagon;
//     arr<arr<dbl, 4>, 4> values;
//     dbl angle_delta = 3.14159265359 / 4.0;
//     dbl angle = 0.0;
//     dbl r = 1.0; r /= 10.0;
//     dbl sx = 2.0;
//     dbl sy = 0.0;
// 
//     for (auto& piace : octagon)
//     // for (st i = 0; i < 4; ++i)
//     {
//         piace[0].x() = r * cos(angle) + sx;
//         piace[0].y() = r * sin(angle) + sy;
//         angle += angle_delta;
// 
//         piace[1].x() = r * cos(angle) + sx;
//         piace[1].y() = r * sin(angle) + sy;
//         angle += angle_delta;
// 
//         piace[2].x() = r * cos(angle) + sx;
//         piace[2].y() = r * sin(angle) + sy;
// 
//         piace[3].x() = 0.0 + sx;
//         piace[3].y() = 0.0 + sy;
//     };
//     octagon[0][0].x() += r / 1.0;
//     octagon[3][2].x() += r / 1.0;
//     octagon[0][0].y() += r / 2.0;
//     octagon[3][2].y() += r / 2.0;
// 
//     // octagon[3][1].x() -= r / 2.0;
// 
//     // octagon[0][0] = prmt::point<2>(-2.0*r + sx, -1.0*r + sy);
//     // octagon[0][1] = prmt::point<2>(-1.0*r + sx, -1.0*r + sy);
//     // octagon[0][2] = prmt::point<2>(0.0*r + sx, 0.0*r + sy);
//     // octagon[0][3] = prmt::point<2>(-2.0*r + sx, 0.0*r + sy);
// 
//     // octagon[1][0] = prmt::point<2>(-1.0*r + sx, -1.0*r + sy);
//     // octagon[1][1] = prmt::point<2>(2.0*r + sx, -1.0*r + sy);
//     // octagon[1][2] = prmt::point<2>(2.0*r + sx, 0.0*r + sy);
//     // octagon[1][3] = prmt::point<2>(0.0*r + sx, 0.0*r + sy);
// 
//     // octagon[2][0] = prmt::point<2>(-2.0*r + sx, 0.0*r + sy);
//     // octagon[2][1] = prmt::point<2>(0.0*r + sx, 0.0*r + sy);
//     // octagon[2][2] = prmt::point<2>(1.0*r + sx, 1.0*r + sy);
//     // octagon[2][3] = prmt::point<2>(-2.0*r + sx, 1.0*r + sy);
// 
//     // octagon[3][0] = prmt::point<2>(0.0*r + sx, 0.0*r + sy);
//     // octagon[3][1] = prmt::point<2>(2.0*r + sx, 0.0*r + sy);
//     // octagon[3][2] = prmt::point<2>(2.0*r + sx, 1.0*r + sy);
//     // octagon[3][3] = prmt::point<2>(1.0*r + sx, 1.0*r + sy);
// 
// 
//     dbl rx = 1.0; rx /= 4.0;
//     dbl ry = 1.0 / 4.0; ry /= 4.0;
// 
//     octagon[0][0] = prmt::point<2>(-2.0*rx + sx, 0.0*ry + sy);
//     octagon[0][1] = prmt::point<2>(-1.0*rx + sx, -1.0*ry + sy);
//     octagon[0][2] = prmt::point<2>(0.0*rx + sx, 0.0*ry + sy);
//     octagon[0][3] = prmt::point<2>(-1.0*rx + sx, 1.0*ry + sy);
// 
//     octagon[1][0] = prmt::point<2>(-1.0*rx + sx, -1.0*ry + sy);
//     octagon[1][1] = prmt::point<2>(-0.5*rx + sx, -2.0*ry + sy);
//     octagon[1][2] = prmt::point<2>(1.0*rx + sx, -1.0*ry + sy);
//     octagon[1][3] = prmt::point<2>(0.0*rx + sx, 0.0*ry + sy);
// 
//     octagon[2][0] = prmt::point<2>(1.0*rx + sx, -1.0*ry + sy);
//     octagon[2][1] = prmt::point<2>(2.0*rx + sx, 0.0*ry + sy);
//     octagon[2][2] = prmt::point<2>(1.0*rx + sx, 1.0*ry + sy);
//     octagon[2][3] = prmt::point<2>(0.0*rx + sx, 0.0*ry + sy);
// 
//     octagon[3][0] = prmt::point<2>(0.0*rx + sx, 0.0*ry + sy);
//     octagon[3][1] = prmt::point<2>(1.0*rx + sx, 1.0*ry + sy);
//     octagon[3][2] = prmt::point<2>(-0.5*rx + sx, 2.0*ry + sy);
//     octagon[3][3] = prmt::point<2>(-1.0*rx + sx, 1.0*ry + sy);
// 
//     // octagon[2][2].y() -= r / 1.0;
//     // octagon[3][2].x() += r / 1.0;
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
//     arr<scalar4pointsfunc<2>, 4> foo = {
//         scalar4pointsfunc<2>(octagon[0], values[0]),
//         scalar4pointsfunc<2>(octagon[1], values[1]),
//         scalar4pointsfunc<2>(octagon[2], values[2]),
//         scalar4pointsfunc<2>(octagon[3], values[3])
//     };
// 
//     file* f;
//     f = fopen("test_iso.gpd", "w");
//     for (st i = 0; i < 4; ++i)
//     {
//         for (st j = 0; j < 4; ++j)
//         {
//             fprintf(f, "%lf %lf %lf %lf %lf\n",
//                     octagon[i][j].x(), octagon[i][j].y(), values[i][j],
//                     foo[i](octagon[i][j]), foo[i].dx(octagon[i][j]));
//         };
//     };
//     fclose(f);
// 
//     // prmt::point<2> cp(0.0 + sx, 0.0 + sy);
//     prmt::point<2> cp(-1.0*rx + sx, -1.0*ry + sy);
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
//             foo[1](-1.0*rx + sx, -1.0*ry + sy),
//             sx*sx*sx+2.0
//             );
// };
// 
// // {
// //     file* f;
// //     f = fopen("test_iso_2.gpd", "w");
// //     dbl r = 1.0;
// //     dbl sx = 1.0;
// //     dbl sy = 0.0;
// //     dbl n = 10;
// //     vec<dbl> angls(n);
// //     for (st i = 0; i < n; ++i)
// //     {
// //         angls[i] = (2.0 * 3.14159265359 / n) * i;
// //     };
// //     for (st i = 0; i < n - 3; ++i)
// //     {
// //         for (st j = i + 1; j < n - 2; ++j)
// //         {
// //             for (st k = j + 1; k < n - 1; ++k)
// //             {
// //                 for (st l = k + 1; l < n; ++l)
// //                 {
// //     arr<prmt::point<2>, 4> p = {
// //         prmt::point<2>(r * cos(angls[i]) + sx, r * sin(angls[i]) + sy),
// //         prmt::point<2>(r * cos(angls[j]) + sx, r * sin(angls[j]) + sy),
// //         prmt::point<2>(r * cos(angls[k]) + sx, r * sin(angls[k]) + sy),
// //         prmt::point<2>(r * cos(angls[l]) + sx, r * sin(angls[l]) + sy)};
// //     arr<dbl, 4> v = {
// //         p[0].x()*p[0].x(),
// //         p[1].x()*p[1].x(),
// //         p[2].x()*p[2].x(),
// //         p[3].x()*p[3].x()};
// //     scalar4pointsfunc<2> foo(p, v);
// //     fprintf(f, "%ld%ld%ld%ld %f %f %f %f %f %f %f %f %f %f %f %f\n",
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
// //     fclose(f);
// // };
// {
//     arr<prmt::point<2>, 4> p = {
//         // prmt::point<2>(0.0, 0.1),
//         // prmt::point<2>(0.0125, 0.0875),
//         // prmt::point<2>(0.040104, 0.096892),
//         // prmt::point<2>(0.031250, 0.109109)
//         prmt::point<2>(0.0, 0.0),
//         prmt::point<2>(2.0, 0.0),
//         prmt::point<2>(2.0, 2.0),
//         prmt::point<2>(1.0, 2.0)
//     };
//     arr<dbl, 4> v = {
//         p[0].x() * p[0].x(),
//         p[1].x() * p[1].x(),
//         p[2].x() * p[2].x(),
//         p[3].x() * p[3].x()
//         // 0.0, 0.000161, 0.000912, 0.001574
//     };
//     scalar4pointsfunc<2> foo(p, v);
//     beeline foo1(p, v);
//     printf("\n");
//     // printf("aaa f=%f v=%f dif=%f s=%f r=%f cs=%f cr=%f\n", 
//     //         foo(p[2]), 
//     //         v[2], 
//     //         foo(p[2]) - v[2], 
//     //         foo.s(p[0].x(), p[0].y()),
//     //         foo.r(p[0].x(), p[0].y()),
//     //         foo.s.c(p[0].x(), p[0].y()),
//     //         foo.r.c(p[0].x(), p[0].y()));
//     cst n = 3;
//     // printf("aaa f=%f v=%f dif=%f s=%f r=%f %f\n", 
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
//     // scalar4pointsfunc<2>::linfunc(1.0, 2.0, 3.0);
//     // puts("!!!!!!!!");
//     // printf("%f\n", foo.r(p[0].x(), p[0].y())[1]);
//     file* f;
//     f = fopen("test_iso_dy.gpd", "w");
//     st nn = 20;
//     dbl summ = 0.0;
//     for (st i = 0; i < nn+1; ++i)
//     {
//         for (st j = 0; j < nn+1; ++j)
//         {
//             dbl x = (p[1].x() - p[0].x()) / nn * i + p[0].x();
//             dbl y = (p[3].y() - p[0].y()) / nn * j + p[0].y();
//             fprintf(f, "%f %f %f %f %f %f %f %f\n", 
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
//     printf("summ %f\n", summ / ((nn+1)*(nn+1)));
//     fclose(f);
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
//     arr<prmt::point<2>, 4> p = {
//         prmt::point<2>(0.0, 0.0),
//         prmt::point<2>(2.0, 0.0),
//         prmt::point<2>(2.0, 2.0),
//         prmt::point<2>(1.0, 2.0)
//     };
//     arr<dbl, 4> v = {
//         p[0].x() * p[0].x(),
//         p[1].x() * p[1].x(),
//         p[2].x() * p[2].x(),
//         p[3].x() * p[3].x()
//     };
//     scalar4pointsfunc<2> foo1(p, v);
//     beeline foo2(p, v);
//     printf("%f %f\n", foo1.dy(1.0, 1.0), foo2.dy(1.0, 1.0));
// };

    // if (0)
    // {
    //     systemslinearalgebraicequations slae1;
    //     systemslinearalgebraicequations slae2;
    //     vec<dealii::Point<2>> v1;
    //     vec<dealii::Point<2>> v2;
    //     {
    //     domain<2> domain;
    //     {
    //         
    //     // set_quadrate<2> (domain.grid, 
    //     //             0.0, 1.0/3.0, 2.0/3.0, 1.0, 
    //     //             -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
    //     //             5);
    //         // set_quadrate<2> (domain.grid, 
    //         //         0.0, 1.0/3.0, 2.0/4.0, 1.0, 
    //         //         -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
    //         //         4);
    //         vec<prmt::point<2>> inner_border;
    //         vec<prmt::point<2>> outer_border;
    //         give_rectangle(inner_border, 1,
    //                 prmt::Point<2>(0.25, -0.25), prmt::Point<2>(0.75, 0.25));
    //         give_rectangle(outer_border, 1,
    //                 prmt::Point<2>(0.0, -0.5), prmt::Point<2>(1.0, 0.5));
    //         set_grid(domain.grid, outer_border, inner_border);
    //         // domain.grid.refine_global(1);
    //     }
    //     debputs();
    //     dealii::fe_q<2> fe(1);
    //     domain.dof_init (fe);
    //
    //     atools ::trivial_prepare_system_equations (slae1, domain);
    //
    //     laplacianscalar<2> element_matrix (domain.dof_handler.get_fe());
    //     {
    //         element_matrix.c .resize(2);
    //         element_matrix.c[0][x][x] = 0.4;
    //         element_matrix.c[0][x][y] = 0.0;
    //         element_matrix.c[0][y][x] = 0.0;
    //         element_matrix.c[0][y][y] = 0.4;
    //         element_matrix.c[1][x][x] = 0.4;
    //         element_matrix.c[1][x][y] = 0.0;
    //         element_matrix.c[1][y][x] = 0.0;
    //         element_matrix.c[1][y][y] = 0.4;
    //     };
    //
    //     // t1.2
    //     cdbl c0 = 0.5;
    //     cdbl e = 1.0;
    //     cdbl nu = 0.25;
    //     cdbl mu = 0.4;
    //     vec<arr<typename nikola::sourcescalar<2>::func, 2>> u(2);
    //     u[0][x] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));}; //ux
    //     u[0][y] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*(p(0)-c0)*p(1);}; //uy
    //     u[1][x] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));};
    //     u[1][y] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*(p(0)-c0)*p(1);};
    //     // u[0][x] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;}; //ux
    //     // u[0][y] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;}; //uy
    //     // u[1][x] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // u[1][y] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // u[0][x] = [] (const dealii::point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;}; //ux
    //     // u[0][y] = [] (const dealii::point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);}; //uy
    //     // u[1][x] = [] (const dealii::point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;};
    //     // u[1][y] = [] (const dealii::point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);};
    //     vec<typename nikola::sourcescalar<2>::func> tau(2);
    //     tau[0] = [e, c0] (const dealii::point<2> &p) {return e*(p(0)-c0);};
    //     tau[1] = [e, c0] (const dealii::point<2> &p) {return e*(p(0)-c0);};
    //     // tau[0] = [e, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // tau[1] = [e, c0] (const dealii::point<2> &p) {return 0.0;};
    //
    //     nikola::sourcescalar<2> element_rhsv (u, tau, domain.dof_handler.get_fe());
    //     // auto func = [] (const dealii::point<2> p) {return -1.0*(p(0)-0.5);};
    //     // sourcescalar<2> element_rhsv (func, domain.dof_handler.get_fe());
    //
    //     assembler::assemble_matrix<2> (slae1.matrix, element_matrix, domain.dof_handler);
    //     assembler::assemble_rhsv<2> (slae1.rhsv, element_rhsv, domain.dof_handler);
    //
    //     dealii::solvercontrol solver_control (10000, 1e-12);
    //     dealii::solvercg<> solver (solver_control);
    //     solver.solve (
    //             slae1.matrix,
    //             slae1.solution,
    //             slae1.rhsv
    //             ,dealii::preconditionidentity()
    //             );
    //
    //     // dbl integral = 0.0;
    //     // dbl area_of_domain = 0.0;
    //     // {
    //     //     dealii::qgauss<2>  quadrature_formula(2);
    //
    //     //     dealii::fevalues<2> fe_values (domain.dof_handler.get_fe(), quadrature_formula,
    //     //             dealii::update_quadrature_points | dealii::update_jxw_values |
    //     //             dealii::update_values);
    //
    //     //     cst n_q_points = quadrature_formula.size();
    //
    //
    //     //     auto cell = domain.dof_handler.begin_active();
    //     //     auto endc = domain.dof_handler.end();
    //     //     for (; cell != endc; ++cell)
    //     //     {
    //     //         fe_values .reinit (cell);
    //
    //     //         dbl area_of_cell = 0.0;
    //     //         for (st q_point = 0; q_point < n_q_points; ++q_point)
    //     //             area_of_cell += fe_values.jxw(q_point);
    //
    //     //         dealii::point<2> c_point(
    //     //                 (cell->vertex(0)(0) +
    //     //                  cell->vertex(1)(0) +
    //     //                  cell->vertex(2)(0) +
    //     //                  cell->vertex(3)(0)) / 4.0,
    //     //                 (cell->vertex(0)(1) +
    //     //                  cell->vertex(1)(1) +
    //     //                  cell->vertex(2)(1) +
    //     //                  cell->vertex(3)(1)) / 4.0);
    //     //         integral += get_value<2> (cell, slae1.solution, c_point) * area_of_cell; 
    //
    //     //         area_of_domain += area_of_cell;
    //     //     };
    //     // };
    //     // printf("%f %f %f\n", integral, area_of_domain, integral / area_of_domain);
    //     dbl integral = 0.0;
    //     dbl area_of_domain = 0.0;
    //     dealii::vector<dbl> s_values(slae1.solution.size());
    //     s_values = 0.0;
    //     {
    //         dealii::qgauss<2>  quadrature_formula(2);
    //
    //         dealii::fevalues<2> fe_values (domain.dof_handler.get_fe(), quadrature_formula,
    //                 dealii::update_quadrature_points | dealii::update_jxw_values |
    //                 dealii::update_values);
    //
    //         cst dofs_per_cell = fe.dofs_per_cell;
    //         cst n_q_points = quadrature_formula.size();
    //         dealii::vector<dbl> cell_value (dofs_per_cell);
    //         vec<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
    //
    //
    //         auto cell = domain.dof_handler.begin_active();
    //         auto endc = domain.dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             fe_values .reinit (cell);
    //             cell_value = 0.0;
    //
    //             for (st i = 0; i < dofs_per_cell; ++i)
    //                 for (st q_point = 0; q_point < n_q_points; ++q_point)
    //                     cell_value[i] += fe_values.shape_value (i, q_point) *
    //                         fe_values.jxw(q_point);
    //
    //             cell->get_dof_indices (local_dof_indices);
    //             for (st i = 0; i < dofs_per_cell; ++i)
    //                 s_values(local_dof_indices[i]) += cell_value(i);
    //         };
    //     };
    //     for (st i = 0; i < slae1.solution.size(); ++i)
    //     {
    //         integral += slae1.solution(i) * s_values(i);
    //         // printf("%.10f %.10f %.10f\n", slae1.solution(i) , s_values(i), slae1.solution(i) * s_values(i));
    //     };
    //     printf("integral %.10f\n", integral);
    //
    //     for (st i = 0; i < slae1.solution.size(); ++i)
    //     {
    //         slae1.solution(i) -= 
    //             integral;
    //             // -0.01480835;
    //             // -0.01253845;
    //     };
    //     printf("\n");
    //
    //
    //
    //     dealii::vector<dbl> uber(slae1.solution.size());
    //     dealii::vector<dbl> diff(slae1.solution.size());
    //     v1.resize(slae1.solution.size());
    //     for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
    //     {
    //         for (st i = 0; i < dealii::geometryinfo<2>::vertices_per_cell; ++i)
    //         {
    //             dbl indx = cell->vertex_dof_index(i, 0);
    //             uber(indx) = uber_function(cell->vertex(i), 200);
    //             diff(indx) = 
    //                 std::abs(uber(indx) - slae1.solution(indx));
    //             v1[indx] = cell->vertex(i);
    //         };
    //     };
    //
    //     integral = 0.0;
    //     for (st i = 0; i < slae1.solution.size(); ++i)
    //     {
    //         integral += slae1.solution(i) * s_values(i);
    //         // printf("(%f,%f) %.10f %.10f %.10f\n", v1[i](0), v1[i](1),
    //         //         slae1.solution(i) , s_values(i), slae1.solution(i) * s_values(i));
    //     };
    //     printf("integral %f\n", integral);
    //
    //     hcptools ::print_temperature<2> (slae1.solution, domain.dof_handler, "temperature-1.gpd");
    //     hcptools ::print_temperature<2> (diff, domain.dof_handler, "uber-diff-1.gpd");
    //     hcptools ::print_temperature<2> (uber, domain.dof_handler, "uber-1.gpd");
    // };        
    //     {
    //     domain<2> domain;
    //     {
    //         
    //     set_quadrate<2> (domain.grid, 
    //                 0.0, 1.0/3.0, 2.0/3.0, 1.0, 
    //                 -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
    //                 4);
    //         // set_quadrate<2> (domain.grid, 
    //         //         0.0, 1.0/3.0, 2.0/4.0, 1.0, 
    //         //         -0.5, 1.0/3.0-0.5, 2.0/3.0-0.5, 0.5,
    //         //         4);
    //     }
    //     debputs();
    //     dealii::fe_q<2> fe(1);
    //     domain.dof_init (fe);
    //
    //     atools ::trivial_prepare_system_equations (slae2, domain);
    //
    //     laplacianscalar<2> element_matrix (domain.dof_handler.get_fe());
    //     {
    //         element_matrix.c .resize(2);
    //         element_matrix.c[0][x][x] = 0.4;
    //         element_matrix.c[0][x][y] = 0.0;
    //         element_matrix.c[0][y][x] = 0.0;
    //         element_matrix.c[0][y][y] = 0.4;
    //         element_matrix.c[1][x][x] = 0.4;
    //         element_matrix.c[1][x][y] = 0.0;
    //         element_matrix.c[1][y][x] = 0.0;
    //         element_matrix.c[1][y][y] = 0.4;
    //     };
    //
    //     // t1.2
    //     cdbl c0 = 0.5;
    //     cdbl e = 1.0;
    //     cdbl nu = 0.25;
    //     cdbl mu = 0.4;
    //     vec<arr<typename nikola::sourcescalar<2>::func, 2>> u(2);
    //     u[0][x] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));}; //ux
    //     u[0][y] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*(p(0)-c0)*p(1);}; //uy
    //     u[1][x] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*0.5*(std::pow(p(0)-c0,2.0)-std::pow(p(1),2.0));};
    //     u[1][y] = [mu, nu, c0] (const dealii::point<2> &p) {return mu*nu*(p(0)-c0)*p(1);};
    //     // u[0][x] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;}; //ux
    //     // u[0][y] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;}; //uy
    //     // u[1][x] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // u[1][y] = [mu, nu, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // u[0][x] = [] (const dealii::point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;}; //ux
    //     // u[0][y] = [] (const dealii::point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);}; //uy
    //     // u[1][x] = [] (const dealii::point<2> &p) {return (p(0)*p(0)-p(1)*p(1))*0.25/2.0*0.4 - 1.0 * p(0) * p(0) / 2.0;};
    //     // u[1][y] = [] (const dealii::point<2> &p) {return 0.25*p(0)*p(1) - 1.0 * p(1) * p(0);};
    //     vec<typename nikola::sourcescalar<2>::func> tau(2);
    //     tau[0] = [e, c0] (const dealii::point<2> &p) {return e*(p(0)-c0);};
    //     tau[1] = [e, c0] (const dealii::point<2> &p) {return e*(p(0)-c0);};
    //     // tau[0] = [e, c0] (const dealii::point<2> &p) {return 0.0;};
    //     // tau[1] = [e, c0] (const dealii::point<2> &p) {return 0.0;};
    //
    //     nikola::sourcescalar<2> element_rhsv (u, tau, domain.dof_handler.get_fe());
    //     // auto func = [] (const dealii::point<2> p) {return -1.0*(p(0)-0.5);};
    //     // sourcescalar<2> element_rhsv (func, domain.dof_handler.get_fe());
    //
    //     assembler::assemble_matrix<2> (slae2.matrix, element_matrix, domain.dof_handler);
    //     assembler::assemble_rhsv<2> (slae2.rhsv, element_rhsv, domain.dof_handler);
    //
    //     dealii::solvercontrol solver_control (10000, 1e-12);
    //     dealii::solvercg<> solver (solver_control);
    //     solver.solve (
    //             slae2.matrix,
    //             slae2.solution,
    //             slae2.rhsv
    //             ,dealii::preconditionidentity()
    //             );
    //
    //     dbl integral = 0.0;
    //     dbl area_of_domain = 0.0;
    //     dealii::vector<dbl> s_values(slae2.solution.size());
    //     s_values = 0.0;
    //     {
    //         dealii::qgauss<2>  quadrature_formula(2);
    //
    //         dealii::fevalues<2> fe_values (domain.dof_handler.get_fe(), quadrature_formula,
    //                 dealii::update_quadrature_points | dealii::update_jxw_values |
    //                 dealii::update_values);
    //
    //         cst dofs_per_cell = fe.dofs_per_cell;
    //         cst n_q_points = quadrature_formula.size();
    //         dealii::vector<dbl> cell_value (dofs_per_cell);
    //         vec<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
    //
    //
    //         auto cell = domain.dof_handler.begin_active();
    //         auto endc = domain.dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             fe_values .reinit (cell);
    //             cell_value = 0.0;
    //
    //             // dbl area_of_cell = 0.0;
    //             // for (st q_point = 0; q_point < n_q_points; ++q_point)
    //             //     area_of_cell += fe_values.jxw(q_point);
    //             for (st i = 0; i < dofs_per_cell; ++i)
    //                 for (st q_point = 0; q_point < n_q_points; ++q_point)
    //                     cell_value[i] += fe_values.shape_value (i, q_point) *
    //                         fe_values.jxw(q_point);
    //
    //             cell->get_dof_indices (local_dof_indices);
    //             for (st i = 0; i < dofs_per_cell; ++i)
    //                 s_values(local_dof_indices[i]) += cell_value(i);
    //             // dealii::point<2> c_point(
    //             //         (cell->vertex(0)(0) +
    //             //          cell->vertex(1)(0) +
    //             //          cell->vertex(2)(0) +
    //             //          cell->vertex(3)(0)) / 4.0,
    //             //         (cell->vertex(0)(1) +
    //             //          cell->vertex(1)(1) +
    //             //          cell->vertex(2)(1) +
    //             //          cell->vertex(3)(1)) / 4.0);
    //             // integral += get_value<2> (cell, slae2.solution, c_point) * area_of_cell; 
    //
    //             // area_of_domain += area_of_cell;
    //         };
    //     };
    //     // printf("%f %f %f\n", integral, area_of_domain, integral / area_of_domain);
    //
    //     for (st i = 0; i < slae2.solution.size(); ++i)
    //     {
    //         integral += slae2.solution(i) * s_values(i);
    //         // printf("%f %f %f\n", slae2.solution(i) , s_values(i), slae2.solution(i) * s_values(i));
    //     };
    //     printf("integral %f\n", integral);
    //
    //     for (st i = 0; i < slae2.solution.size(); ++i)
    //     {
    //         slae2.solution(i) -= integral;
    //     };
    //     dealii::vector<dbl> uber(slae2.solution.size());
    //     dealii::vector<dbl> diff(slae2.solution.size());
    //     v2.resize(slae2.solution.size());
    //     for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
    //     {
    //         for (st i = 0; i < dealii::geometryinfo<2>::vertices_per_cell; ++i)
    //         {
    //             dbl indx = cell->vertex_dof_index(i, 0);
    //             uber(indx) = uber_function(cell->vertex(i), 200);
    //             diff(indx) = 
    //                 std::abs(uber(indx) - slae2.solution(indx));
    //             v2[indx] = cell->vertex(i);
    //         };
    //     };
    //
    //     hcptools ::print_temperature<2> (slae2.solution, domain.dof_handler, "temperature-2.gpd");
    //     hcptools ::print_temperature<2> (diff, domain.dof_handler, "uber-diff-2.gpd");
    // };
    //     
    //     // dealii::vector<dbl> uber(slae1.solution.size());
    //     // dealii::vector<dbl> diff(slae1.solution.size());
    //     // for (auto cell = domain.dof_handler.begin_active (); cell != domain.dof_handler.end (); ++cell)
    //     // {
    //     //     for (st i = 0; i < dealii::geometryinfo<2>::vertices_per_cell; ++i)
    //     //     {
    //     //         dbl indx = cell->vertex_dof_index(i, 0);
    //     //         uber(indx) = uber_function(cell->vertex(i), 200);
    //     //         diff(indx) = 
    //     //             // std::abs(uber(indx) - slae.solution(indx));
    //     //          slae2.solution(indx) - slae1.solution(indx);
    //     //     };
    //     // };
    //     file *f;
    //     f = fopen("diff.gpd", "w");
    //     for (st i = 0; i < v1.size(); ++i)
    //     {
    //         cdbl s1 = slae1.solution(i);
    //         dbl s2 = 0.0;
    //         // printf("(%f %f) ", v1[i](0), v1[i](1));
    //         for (st j = 0; j < v2.size(); ++j)
    //         {
    //             // printf("%f\n",v1[i].distance(v2[j]));
    //             if (v1[i].distance(v2[j]) < 1.0e-10)
    //             {
    //                 s2 = slae2.solution(j);
    //                 // printf("(%f %f)\n", v2[j](0), v2[j](1));
    //                 fprintf(f, "%f %f %f\n", v1[i](0), v1[i](1), std::abs(s1 - s2));
    //                 break;
    //             };
    //         };
    //     };
    //     fclose(f);
    //
    //     // hcptools ::print_temperature<2> (slae1.solution, domain.dof_handler, "temperature-3.gpd");
    //     // hcptools ::print_temperature<2> (uber, domain.dof_handler, "uber.gpd");
    //     // hcptools ::print_temperature<2> (diff, domain.dof_handler, "uber-diff-2.gpd");
    // };
    return EXIT_SUCCESS;
}
