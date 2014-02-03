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

int main()
{
    enum {x, y};
    debputs();

    Domain<2> domain;
    {
        vec<prmt::Point<2>> border;
        vec<st> type_border;
        arr<st, 4> types = {0, 0, 0, 0};
        prmt::Point<2> p1(0.0, 0.0);
        prmt::Point<2> p2(1.0, 1.0);
    debputs();
        GTools::give_rectangle_with_border_condition (
                border, type_border, types, 15, p1, p2);
    debputs();
        make_grid (domain.grid, border, type_border);
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
    
    arr<prmt::Point<2>, 4> points = {
        prmt::Point<2>(0.0, 0.0),
        prmt::Point<2>(1.0, 0.0),
        prmt::Point<2>(1.0, 1.0),
        prmt::Point<2>(0.0, 1.0)};
    arr<dbl, 4> values = {
        0.0,
        1.0,
        1.0,
        0.0};

    Scalar4PointsFunc<2> f1(points, values);

    points[0] = prmt::Point<2>(1.0, 0.0);
    points[1] = prmt::Point<2>(1.0, 0.0);
    points[2] = prmt::Point<2>(2.0, 1.0);
    points[3] = prmt::Point<2>(1.0, 1.0);

    values[0] = 1.0;
    values[1] = 4.0;
    values[2] = 4.0;
    values[3] = 1.0;

    Scalar4PointsFunc<2> f2(points, values);

    printf("%lf\n", f1(0.0, 0.0));
    printf("%lf\n", f1(1.0, 1.0));
    printf("%lf\n", f1.dx(1.0, 1.0));
    printf("%lf\n", f2(1.0, 1.0));
    printf("%lf\n", f2.dx(1.0, 1.0));

    return EXIT_SUCCESS;
}
