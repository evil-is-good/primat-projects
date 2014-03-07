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

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

extern void make_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<st>);

void debputs()
{
    static int n = 0;
    printf("%d\n", n);
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

int main()
{
    enum {x, y};
    debputs();

    Domain<2> domain;
    {
        vec<prmt::Point<2>> boundary_of_segments;
        vec<st> types_boundary_segments;
        arr<st, 4> types_boundary = {0, 0, 0, 0};
        cst num_segments = 2;
        prmt::Point<2> p1(0.0, 0.0);
        prmt::Point<2> p2(1.0, 1.0);
    debputs();
        GTools::give_rectangle_with_border_condition (
                boundary_of_segments, types_boundary_segments, 
                types_boundary, num_segments, p1, p2);
    debputs();
        make_grid (domain.grid, boundary_of_segments, types_boundary_segments);
        domain.grid.refine_global(1);
    };
    debputs();
    dealii::FE_Q<2> fe(1);
    domain.dof_init (fe);

    SystemsLinearAlgebraicEquations slae;
    ATools ::trivial_prepare_system_equations (slae, domain);

    LaplacianScalar<2> element_matrix (domain.dof_handler.get_fe());
    {
        arr<arr<vec<dbl>, 2>, 2> coef;
        coef[x][x] .push_back (1.0);
        coef[y][y] .push_back (1.0);
        coef[x][y] .push_back (0.0);
        coef[y][x] .push_back (0.0);
        HCPTools ::set_thermal_conductivity<2> (element_matrix.C, coef);  
    };

    auto func = [] (dealii::Point<2>) {return -2.0;};
    SourceScalar<2> element_rhsv (func, domain.dof_handler.get_fe());

    Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_handler);
    Assembler::assemble_rhsv<2> (slae.rhsv, element_rhsv, domain.dof_handler);

    vec<BoundaryValueScalar<2>> bound (1);
    bound[0].function      = [] (const dealii::Point<2> &p) {return p(0) * p(0);};
    bound[0].boundary_id   = 0;
    bound[0].boundary_type = TBV::Dirichlet;

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

    HCPTools ::print_temperature<2> (slae.solution, domain.dof_handler, "temperature");
    HCPTools ::print_heat_conductions<2> (
            slae.solution, element_matrix.C, domain, "heat_conductions");
    HCPTools ::print_heat_gradient<2> (
            slae.solution, element_matrix.C, domain, "heat_gradient");
    
    {
    arr<prmt::Point<2>, 4> points1 = {
        prmt::Point<2>(0.0, 0.0),
        prmt::Point<2>(2.0, 0.0),
        prmt::Point<2>(1.5, 1.5),
        prmt::Point<2>(0.0, 2.0)};
    arr<dbl, 4> values1 = {
        points1[0].x()*points1[0].x(),
        points1[1].x()*points1[1].x(),
        points1[2].x()*points1[2].x(),
        points1[3].x()*points1[3].x()};

    arr<prmt::Point<2>, 4> points2 = {
        prmt::Point<2>(2.0, 0.0),
        prmt::Point<2>(4.0, 3.0),
        prmt::Point<2>(3.0, 4.0),
        prmt::Point<2>(1.5, 1.5)};
    arr<dbl, 4> values2 = {
        points2[0].x()*points2[0].x(),
        points2[1].x()*points2[1].x(),
        points2[2].x()*points2[2].x(),
        points2[3].x()*points2[3].x()};
    // printf("%lf %lf %lf %lf\n", 
    //         values[0],
    //         values[1],
    //         values[2],
    //         values[3]);

    Scalar4PointsFunc<2> f1(points1, values1);

    // points[0] = prmt::Point<2>(1.0, 0.0);
    // points[1] = prmt::Point<2>(2.0, 0.0);
    // points[2] = prmt::Point<2>(3.0, 1.0);
    // points[3] = prmt::Point<2>(2.0, 1.0);

    // values[0] = points[0].x()*points[0].x();
    // values[1] = points[1].x()*points[1].x();
    // values[2] = points[2].x()*points[2].x();
    // values[3] = points[3].x()*points[3].x();
    // printf("%lf %lf %lf %lf\n", 
    //         values[0],
    //         values[1],
    //         values[2],
    //         values[3]);

    Scalar4PointsFunc<2> f2(points2, values2);

    auto ip = prmt::Point<2>(1.5, 1.5);

    // printf("%lf\n", f1(0.0, 0.0));
    // printf("%lf\n", f1(ip));
    // printf("%lf\n", f1.dx(ip));
    // printf("%lf\n", f2(ip));
    // printf("%lf\n", f2.dx(ip));


    arr<arr<prmt::Point<2>, 4>, 4> octagon;
    arr<arr<dbl, 4>, 4> values;
    dbl angle_delta = 3.14159265359 / 4.0;
    dbl angle = 0.0;
    dbl R = 1.0; R /= 10.0;
    dbl sx = 2.0;
    dbl sy = 0.0;

    for (auto& piace : octagon)
    // for (st i = 0; i < 4; ++i)
    {
        piace[0].x() = R * cos(angle) + sx;
        piace[0].y() = R * sin(angle) + sy;
        angle += angle_delta;

        piace[1].x() = R * cos(angle) + sx;
        piace[1].y() = R * sin(angle) + sy;
        angle += angle_delta;

        piace[2].x() = R * cos(angle) + sx;
        piace[2].y() = R * sin(angle) + sy;

        piace[3].x() = 0.0 + sx;
        piace[3].y() = 0.0 + sy;
    };
    octagon[0][0].x() += R / 1.0;
    octagon[3][2].x() += R / 1.0;
    octagon[0][0].y() += R / 2.0;
    octagon[3][2].y() += R / 2.0;

    // octagon[3][1].x() -= R / 2.0;

    // octagon[0][0] = prmt::Point<2>(-2.0*R + sx, -1.0*R + sy);
    // octagon[0][1] = prmt::Point<2>(-1.0*R + sx, -1.0*R + sy);
    // octagon[0][2] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
    // octagon[0][3] = prmt::Point<2>(-2.0*R + sx, 0.0*R + sy);

    // octagon[1][0] = prmt::Point<2>(-1.0*R + sx, -1.0*R + sy);
    // octagon[1][1] = prmt::Point<2>(2.0*R + sx, -1.0*R + sy);
    // octagon[1][2] = prmt::Point<2>(2.0*R + sx, 0.0*R + sy);
    // octagon[1][3] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);

    // octagon[2][0] = prmt::Point<2>(-2.0*R + sx, 0.0*R + sy);
    // octagon[2][1] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
    // octagon[2][2] = prmt::Point<2>(1.0*R + sx, 1.0*R + sy);
    // octagon[2][3] = prmt::Point<2>(-2.0*R + sx, 1.0*R + sy);

    // octagon[3][0] = prmt::Point<2>(0.0*R + sx, 0.0*R + sy);
    // octagon[3][1] = prmt::Point<2>(2.0*R + sx, 0.0*R + sy);
    // octagon[3][2] = prmt::Point<2>(2.0*R + sx, 1.0*R + sy);
    // octagon[3][3] = prmt::Point<2>(1.0*R + sx, 1.0*R + sy);


    dbl Rx = 1.0; Rx /= 4.0;
    dbl Ry = 1.0 / 4.0; Ry /= 4.0;

    octagon[0][0] = prmt::Point<2>(-2.0*Rx + sx, 0.0*Ry + sy);
    octagon[0][1] = prmt::Point<2>(-1.0*Rx + sx, -1.0*Ry + sy);
    octagon[0][2] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
    octagon[0][3] = prmt::Point<2>(-1.0*Rx + sx, 1.0*Ry + sy);

    octagon[1][0] = prmt::Point<2>(-1.0*Rx + sx, -1.0*Ry + sy);
    octagon[1][1] = prmt::Point<2>(-0.5*Rx + sx, -2.0*Ry + sy);
    octagon[1][2] = prmt::Point<2>(1.0*Rx + sx, -1.0*Ry + sy);
    octagon[1][3] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);

    octagon[2][0] = prmt::Point<2>(1.0*Rx + sx, -1.0*Ry + sy);
    octagon[2][1] = prmt::Point<2>(2.0*Rx + sx, 0.0*Ry + sy);
    octagon[2][2] = prmt::Point<2>(1.0*Rx + sx, 1.0*Ry + sy);
    octagon[2][3] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);

    octagon[3][0] = prmt::Point<2>(0.0*Rx + sx, 0.0*Ry + sy);
    octagon[3][1] = prmt::Point<2>(1.0*Rx + sx, 1.0*Ry + sy);
    octagon[3][2] = prmt::Point<2>(-0.5*Rx + sx, 2.0*Ry + sy);
    octagon[3][3] = prmt::Point<2>(-1.0*Rx + sx, 1.0*Ry + sy);

    // octagon[2][2].y() -= R / 1.0;
    // octagon[3][2].x() += R / 1.0;

    for (st i = 0; i < 4; ++i)
    {
        for (st j = 0; j < 4; ++j)
        {
            values[i][j] = 
                octagon[i][j].x() * octagon[i][j].x() * octagon[i][j].x() + 2.0
                // + octagon[i][j].y() * octagon[i][j].y()
                ;
        };
    };

    arr<Scalar4PointsFunc<2>, 4> foo = {
        Scalar4PointsFunc<2>(octagon[0], values[0]),
        Scalar4PointsFunc<2>(octagon[1], values[1]),
        Scalar4PointsFunc<2>(octagon[2], values[2]),
        Scalar4PointsFunc<2>(octagon[3], values[3])
    };

    FILE* F;
    F = fopen("test_iso.gpd", "w");
    for (st i = 0; i < 4; ++i)
    {
        for (st j = 0; j < 4; ++j)
        {
            fprintf(F, "%lf %lf %lf %lf %lf\n",
                    octagon[i][j].x(), octagon[i][j].y(), values[i][j],
                    foo[i](octagon[i][j]), foo[i].dx(octagon[i][j]));
        };
    };
    fclose(F);

    // prmt::Point<2> cp(0.0 + sx, 0.0 + sy);
    prmt::Point<2> cp(-1.0*Rx + sx, -1.0*Ry + sy);

    printf("%lf %lf\n", 
            (foo[0].dx(cp) +
            foo[1].dx(cp) +
            foo[2].dx(cp) +
            foo[3].dx(cp)) / 4.0,
            // sx*2.0
            3.0*sx*sx
            );

    printf("%lf %lf\n", 
            foo[1](-1.0*Rx + sx, -1.0*Ry + sy),
            sx*sx*sx+2.0
            );
};

// {
//     FILE* F;
//     F = fopen("test_iso_2.gpd", "w");
//     dbl R = 1.0;
//     dbl sx = 1.0;
//     dbl sy = 0.0;
//     dbl N = 10;
//     vec<dbl> angls(N);
//     for (st i = 0; i < N; ++i)
//     {
//         angls[i] = (2.0 * 3.14159265359 / N) * i;
//     };
//     for (st i = 0; i < N - 3; ++i)
//     {
//         for (st j = i + 1; j < N - 2; ++j)
//         {
//             for (st k = j + 1; k < N - 1; ++k)
//             {
//                 for (st l = k + 1; l < N; ++l)
//                 {
//     arr<prmt::Point<2>, 4> p = {
//         prmt::Point<2>(R * cos(angls[i]) + sx, R * sin(angls[i]) + sy),
//         prmt::Point<2>(R * cos(angls[j]) + sx, R * sin(angls[j]) + sy),
//         prmt::Point<2>(R * cos(angls[k]) + sx, R * sin(angls[k]) + sy),
//         prmt::Point<2>(R * cos(angls[l]) + sx, R * sin(angls[l]) + sy)};
//     arr<dbl, 4> v = {
//         p[0].x()*p[0].x(),
//         p[1].x()*p[1].x(),
//         p[2].x()*p[2].x(),
//         p[3].x()*p[3].x()};
//     Scalar4PointsFunc<2> foo(p, v);
//     fprintf(F, "%ld%ld%ld%ld %f %f %f %f %f %f %f %f %f %f %f %f\n",
//             i, j, k, l, 
//             p[0].x()*p[0].x(),
//             p[1].x()*p[1].x(),
//             p[2].x()*p[2].x(),
//             p[3].x()*p[3].x(),
//             foo(p[0]),
//             foo(p[1]),
//             foo(p[2]),
//             foo(p[3]),
//             (foo(p[0]) - p[0].x()*p[0].x()),
//             (foo(p[1]) - p[1].x()*p[1].x()),
//             (foo(p[2]) - p[2].x()*p[2].x()),
//             (foo(p[3]) - p[3].x()*p[3].x()));
//                     
//                 };
//             };
//         };
//     };
//     fclose(F);
// };
{
    arr<prmt::Point<2>, 4> p = {
        // prmt::Point<2>(0.0, 0.1),
        // prmt::Point<2>(0.0125, 0.0875),
        // prmt::Point<2>(0.040104, 0.096892),
        // prmt::Point<2>(0.031250, 0.109109)
        prmt::Point<2>(0.0, 0.0),
        prmt::Point<2>(2.0, 0.0),
        prmt::Point<2>(2.0, 2.0),
        prmt::Point<2>(1.0, 2.0)
    };
    arr<dbl, 4> v = {
        p[0].x() * p[0].x(),
        p[1].x() * p[1].x(),
        p[2].x() * p[2].x(),
        p[3].x() * p[3].x()
        // 0.0, 0.000161, 0.000912, 0.001574
    };
    Scalar4PointsFunc<2> foo(p, v);
    Beeline foo1(p, v);
    printf("\n");
    // printf("AAA f=%f v=%f dif=%f s=%f r=%f Cs=%f Cr=%f\n", 
    //         foo(p[2]), 
    //         v[2], 
    //         foo(p[2]) - v[2], 
    //         foo.s(p[0].x(), p[0].y()),
    //         foo.r(p[0].x(), p[0].y()),
    //         foo.s.C(p[0].x(), p[0].y()),
    //         foo.r.C(p[0].x(), p[0].y()));
    cst n = 3;
    // printf("AAA f=%f v=%f dif=%f s=%f r=%f %f\n", 
    //         foo(p[n]), 
    //         v[n], 
    //         foo(p[n]) - v[n], 
    //         foo.s(p[n].x(), p[n].y())[0],
    //         foo.r(p[n].x(), p[n].y())[1], std::sqrt(-4.0));
    // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
    //         foo.s(p[0].x(), p[0].y())[0],
    //         foo.r(p[0].x(), p[0].y())[1],
    //         foo.s(p[0].x(), p[0].y())[1],
    //         foo.r(p[0].x(), p[0].y())[0],
    //         foo.s(p[1].x(), p[1].y())[0],
    //         foo.r(p[1].x(), p[1].y())[1],
    //         foo.s(p[1].x(), p[1].y())[1],
    //         foo.r(p[1].x(), p[1].y())[0],
    //         foo.s(p[2].x(), p[2].y())[0],
    //         foo.r(p[2].x(), p[2].y())[1],
    //         foo.s(p[2].x(), p[2].y())[1],
    //         foo.r(p[2].x(), p[2].y())[0],
    //         foo.s(p[3].x(), p[3].y())[0],
    //         foo.r(p[3].x(), p[3].y())[1],
    //         foo.s(p[3].x(), p[3].y())[1],
    //         foo.r(p[3].x(), p[3].y())[0]
    //         );
    // dbl midl_x = (0.0 + 0.0125 + 0.040104 + 0.031250) / 4.0;
    // dbl midl_y = (0.1 + 0.0875 + 0.096892 + 0.109109) / 4.0;
    // Scalar4PointsFunc<2>::LinFunc(1.0, 2.0, 3.0);
    // puts("!!!!!!!!");
    // printf("%f\n", foo.r(p[0].x(), p[0].y())[1]);
    FILE* F;
    F = fopen("test_iso_dy.gpd", "w");
    st NN = 20;
    dbl summ = 0.0;
    for (st i = 0; i < NN+1; ++i)
    {
        for (st j = 0; j < NN+1; ++j)
        {
            dbl x = (p[1].x() - p[0].x()) / NN * i + p[0].x();
            dbl y = (p[3].y() - p[0].y()) / NN * j + p[0].y();
            fprintf(F, "%f %f %f %f %f %f %f %f\n", 
                    x,
                    y,
                    foo(x, y),
                    foo.dy(x, y),
                    foo1(x, y),
                    foo1.dy(x, y),
                    x*x,
                    2*x);
            summ += foo.dy(x, y);
        };
    };
    printf("summ %f\n", summ / ((NN+1)*(NN+1)));
    fclose(F);
    // dbl midl_x = (p[0].x() + p[1].x() + p[2].x() + p[3].x()) / 4.0;  
    // dbl midl_y = (p[0].y() + p[1].y() + p[2].y() + p[3].y()) / 4.0;  
    // printf("%f %f %f %f %f\n", midl_x, midl_y, foo.dy(midl_x, midl_y), foo.dy(0.0, 0.0),
    //         foo(0.0, 0.05));
    // puts("!!!!!!!!");
    // for (st i = 0; i < 11; ++i)
    // {
    //     dbl x = 0.0;
    //     dbl y = -0.02 + i*0.01;
    //     dbl s = foo.s(x,y)[0];
    //     dbl r = foo.r(x,y)[1];
    //     dbl f = foo(x,y);
    //     printf("%f %f %f %f\n", y, s, r, f);
    //     // foo(x, y);
    // };
    // // foo(0.0, 0.0)
    // // foo(0.0, 0.05);
};
{
    arr<prmt::Point<2>, 4> p = {
        prmt::Point<2>(0.0, 0.0),
        prmt::Point<2>(2.0, 0.0),
        prmt::Point<2>(2.0, 2.0),
        prmt::Point<2>(1.0, 2.0)
    };
    arr<dbl, 4> v = {
        p[0].x() * p[0].x(),
        p[1].x() * p[1].x(),
        p[2].x() * p[2].x(),
        p[3].x() * p[3].x()
    };
    Scalar4PointsFunc<2> foo1(p, v);
    Beeline foo2(p, v);
    printf("%f %f\n", foo1.dy(1.0, 1.0), foo2.dy(1.0, 1.0));
};

    return EXIT_SUCCESS;
}
