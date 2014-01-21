#include <stdlib.h>
// #include <projects/deal/tests/elastic_problem/elastic_problem.h>
// #include <projects/deal/tests/heat_conduction_problem/heat_condrction_problem.h>
#include </home/primat/projects/deal/tests/elastic_problem/elastic_problem.h>
#include </home/primat/projects/deal/tests/elastic_problem_2d_on_cell_v2/elastic_problem_2d_on_cell.h>
#include </home/primat/projects/deal/tests/heat_conduction_problem/heat_conduction_problem.h>
#include </home/primat/projects/deal/tests/heat_conduction_problem_on_cell/heat_conduction_problem_on_cell.h>
// #include </home/primat/projects/deal/tests/esm_laplace_problem/esm_laplace_problem.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_reordering.h>
#include "projects/cae/main/point/point.h"
#include "projects/cae/test/file/file.h"

#include "projects/calculation_core/blocks/general/laplacian/scalar/laplacian_scalar.h"
#include "projects/calculation_core/blocks/general/laplacian/vector/laplacian_vector.h"

#include "projects/calculation_core/blocks/general/source/scalar/source_scalar.h"
#include "projects/calculation_core/blocks/general/source/vector/source_vector.h"

#include "projects/calculation_core/blocks/special/problem_on_cell/source/scalar/source_scalar.h"
#include "projects/calculation_core/blocks/special/problem_on_cell/source/vector/source_vector.h"

#include "projects/calculation_core/blocks/general/boundary_value/boundary_value.h"

#include </home/primat/projects/calculation_core/blocks/general/4_points_function/4_points_function.h>

template<uint8_t dim>
arr<dealii::Point<dim, double>, dim> get_grad_elastic (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> solution,
        uint8_t index_vertex)
{
    std::array<dealii::Point<dim, double>, dim> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    FOR_I(0, dim)
    {
        double f1 = solution(cell->vertex_dof_index (0, i));
        double f2 = solution(cell->vertex_dof_index (1, i));
        double f3 = solution(cell->vertex_dof_index (2, i));
        double f4 = solution(cell->vertex_dof_index (3, i));

        double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

        grad[i](0) = b + d * cell->vertex(index_vertex)(1);
        grad[i](1) = c + d * cell->vertex(index_vertex)(0);
    };

    return grad;
};

template<uint8_t dim>
arr<arr<dbl, 2>, 2> get_grad_elastic_on_element (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::FESystem<dim> &fe,
        const dealii::Vector<double> solution)
{
    enum {x, y, z};

    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (fe, quadrature_formula,
            dealii::update_gradients | 
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const uint8_t dofs_per_cell = fe.dofs_per_cell;
    const uint8_t num_quad_points = quadrature_formula.size();

    fe_values .reinit (cell);

    dbl area = 0.0;
    FOR(i, 0, num_quad_points)
        area += fe_values.JxW(i);

    arr<arr<dbl,2>,2> grad;

    for (auto i : {x, y}) 
        for (auto j : {x, y}) 
        {
            grad[i][j] = 0.0;

            FOR(point, 0, 4)
            {
                dbl temp = 0.0;
                FOR(q_point, 0, num_quad_points)
                    temp += 
                        fe_values.shape_grad ((point * 2 + i), q_point)[j] *
                        fe_values.JxW(q_point);
                grad[i][j] += 
                    solution(cell->vertex_dof_index(point, i)) * temp;
            };
            grad[i][j] /= area;
        };

    return grad;
};

template<int dim>
std::array<double, dim> boundary (const dealii::Point<dim> &p)
{
    std::array<double, dim> res;
    for (size_t i = 0; i < dim; ++i)
        res[i] = p(0);//p(0)*p(0)*p(1)*p(1);//p(i)*p(i);
    return res;
//    return (p(0)*p(0))*(p(1)*p(1));
};

template<int dim>
std::array<double, dim> source (const dealii::Point<dim> &p)
{
    std::array<double, dim> res;
//    for (size_t i = 0; i < dim; ++i)
//        res[i] = -6.0;//0.0;
    res[0] = 0.0;// - (6*p(1)*p(1) + 4*p(0)*p(1) + 4*p(0)*p(1) + 2 * p(0)*p(0));
    res[1] = 0.0;//- (6*p(0)*p(0) + 4*p(0)*p(1) + 4*p(0)*p(1) + 2 * p(1)*p(1));
    return res;
//    return -2.0*((p(0)*p(0)) + (p(1)*p(1)));
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


template<uint8_t dim>
void calc_stress_and_deform(const ElasticProblem<dim> &problem,
     const typename ElasticProblemSup<dim>::TypeCoef &C,
     arr<dealii::Vector<dbl>, 2> &stress,
     arr<dealii::Vector<dbl>, 2> &deform)
{
    enum {x, y, z};

    stress[0].reinit(problem.system_equations.x.size());
    stress[1].reinit(problem.system_equations.x.size());

    deform[0].reinit(problem.system_equations.x.size());
    deform[1].reinit(problem.system_equations.x.size());
    
    vec<u8> divider(problem.system_equations.x.size());

    typename dealii::DoFHandler<2>::active_cell_iterator cell =
        problem.domain.dof_handler.begin_active();

    typename dealii::DoFHandler<2>::active_cell_iterator endc =
        problem.domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        FOR(i, 0, 4)
        {
            auto grad = 
                ::get_grad_elastic<dim> (cell, problem.system_equations.x, i);  

            cdbl sigma_xx = 
                C[x][x][x][x][0] * grad[x](x) +
                C[x][x][x][y][0] * grad[x](y) +
                C[x][x][y][x][0] * grad[y](x) +
                C[x][x][y][y][0] * grad[y](y);

            cdbl sigma_xy = 
                C[x][y][x][x][0] * grad[x](x) +
                C[x][y][x][y][0] * grad[x](y) +
                C[x][y][y][x][0] * grad[y](x) +
                C[x][y][y][y][0] * grad[y](y);

            cdbl sigma_yx = 
                C[y][x][x][x][0] * grad[x](x) +
                C[y][x][x][y][0] * grad[x](y) +
                C[y][x][y][x][0] * grad[y](x) +
                C[y][x][y][y][0] * grad[y](y);

            cdbl sigma_yy = 
                C[y][y][x][x][0] * grad[x](x) +
                C[y][y][x][y][0] * grad[x](y) +
                C[y][y][y][x][0] * grad[y](x) +
                C[y][y][y][y][0] * grad[y](y);

            stress[x][cell->vertex_dof_index(i, x)] += sigma_xx;
            stress[x][cell->vertex_dof_index(i, y)] += sigma_xy;
            stress[y][cell->vertex_dof_index(i, x)] += sigma_yx;
            stress[y][cell->vertex_dof_index(i, y)] += sigma_yy;

            deform[x][cell->vertex_dof_index(i, x)] += grad[x][x];
            deform[x][cell->vertex_dof_index(i, y)] += grad[x][y];
            deform[y][cell->vertex_dof_index(i, x)] += grad[y][x];
            deform[y][cell->vertex_dof_index(i, y)] += grad[y][y];

            divider[cell->vertex_dof_index(i,0)] += 1;
            divider[cell->vertex_dof_index(i,1)] += 1;
        };
    };

    FOR(i, 0, divider.size())
    {
        stress[x][i] /= divider[i];
        stress[y][i] /= divider[i];

        deform[x][i] /= divider[i];
        deform[y][i] /= divider[i];
    };
};

template<uint8_t dim>
void print_stress(const ElasticProblem<dim> &problem,
     const typename ElasticProblemSup<dim>::TypeCoef &C,
     const str file_name,
     arr<dealii::Vector<dbl>, 2> &stress)
{
    enum {x, y, z};
    char suffix[3] = {'x', 'y', 'z'};
    // 
    // // vec<dealii::Point<2>> stress(problem.system_equations.x.size());
    // arr<dealii::Vector<dbl>, 2> stress;
    // stress[0].reinit(problem.system_equations.x.size());
    // stress[1].reinit(problem.system_equations.x.size());
    // vec<u8> divider(problem.system_equations.x.size());

    // typename dealii::DoFHandler<2>::active_cell_iterator cell =
    //     problem.domain.dof_handler.begin_active();

    // typename dealii::DoFHandler<2>::active_cell_iterator endc =
    //     problem.domain.dof_handler.end();

    // for (; cell != endc; ++cell)
    // {
    //     FOR(i, 0, 4)
    //     {
    //         auto deform = 
    //             ::get_grad_elastic<dim> (cell, problem.system_equations.x, i);  

    //         cdbl sigma_xx = 
    //             C[x][x][x][x][0] * deform[x](x) +
    //             C[x][x][x][y][0] * deform[x](y) +
    //             C[x][x][y][x][0] * deform[y](x) +
    //             C[x][x][y][y][0] * deform[y](y);
    //         // printf("Cxxyy = %f\n", C[x][x][y][y][0]);

    //         cdbl sigma_xy = 
    //             C[x][y][x][x][0] * deform[x](x) +
    //             C[x][y][x][y][0] * deform[x](y) +
    //             C[x][y][y][x][0] * deform[y](x) +
    //             C[x][y][y][y][0] * deform[y](y);

    //         cdbl sigma_yx = 
    //             C[y][x][x][x][0] * deform[x](x) +
    //             C[y][x][x][y][0] * deform[x](y) +
    //             C[y][x][y][x][0] * deform[y](x) +
    //             C[y][x][y][y][0] * deform[y](y);

    //         cdbl sigma_yy = 
    //             C[y][y][x][x][0] * deform[x](x) +
    //             C[y][y][x][y][0] * deform[x](y) +
    //             C[y][y][y][x][0] * deform[y](x) +
    //             C[y][y][y][y][0] * deform[y](y);

    //         // stress[cell->vertex_dof_index(i,0)] += 
    //         //     // dealii::Point<2>(deform[x](x), deform[x][y]);
    //         //     dealii::Point<2>(sigma_xx, sigma_xy);

    //         // stress[cell->vertex_dof_index(i,1)] += 
    //         //     // dealii::Point<2>(deform[y](x), deform[y][y]);
    //         //     dealii::Point<2>(sigma_yx, sigma_yy);

    //         stress[x][cell->vertex_dof_index(i, x)] += sigma_xx;
    //         stress[x][cell->vertex_dof_index(i, y)] += sigma_xy;
    //         stress[y][cell->vertex_dof_index(i, x)] += sigma_yx;
    //         stress[y][cell->vertex_dof_index(i, y)] += sigma_yy;

    //         divider[cell->vertex_dof_index(i,0)] += 1;
    //         divider[cell->vertex_dof_index(i,1)] += 1;
    //     };
    // };

    // FOR(i, 0, divider.size())
    // {
    //     stress[x][i] /= divider[i];
    //     stress[y][i] /= divider[i];
    // };

    for (auto i : {x,y})
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (problem.domain.dof_handler);

        data_out.add_data_vector (stress[i], "x y");
        data_out.build_patches ();

        std::ofstream output (file_name + "stress_" + suffix[i] + ".gpd");
        data_out.write_gnuplot (output);
    };
};

template<uint8_t dim>
void calc_stress_and_deform_on_element(const ElasticProblem<dim> &problem,
     const typename ElasticProblemSup<dim>::TypeCoef &C,
     vec<arr<arr<dbl, 2>, 2>> &stress,
     vec<arr<arr<dbl, 2>, 2>> &deform,
     vec<prmt::Point<2>> &points)
{
    enum {x, y, z};

    typename dealii::DoFHandler<2>::active_cell_iterator cell =
        problem.domain.dof_handler.begin_active();

    typename dealii::DoFHandler<2>::active_cell_iterator endc =
        problem.domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        auto grad = 
            ::get_grad_elastic_on_element<dim> (
                    cell, problem.finite_element,
                    problem.system_equations.x);  

        arr<arr<dbl, 2>, 2> sigma =
                 {
                    arr<dbl, 2>{
                        C[x][x][x][x][0] * grad[x][x] +
                        C[x][x][x][y][0] * grad[x][y] +
                        C[x][x][y][x][0] * grad[y][x] +
                        C[x][x][y][y][0] * grad[y][y],

                        C[x][y][x][x][0] * grad[x][x] +
                        C[x][y][x][y][0] * grad[x][y] +
                        C[x][y][y][x][0] * grad[y][x] +
                        C[x][y][y][y][0] * grad[y][y]
                    },
                    arr<dbl, 2>{
                        C[y][x][x][x][0] * grad[x][x] +
                        C[y][x][x][y][0] * grad[x][y] +
                        C[y][x][y][x][0] * grad[y][x] +
                        C[y][x][y][y][0] * grad[y][y],

                        C[y][y][x][x][0] * grad[x][x] +
                        C[y][y][x][y][0] * grad[x][y] +
                        C[y][y][y][x][0] * grad[y][x] +
                        C[y][y][y][y][0] * grad[y][y]
                    }
                };

        prmt::Point<2> midle(
                (cell->vertex(0)[0] +
                 cell->vertex(1)[0] +
                 cell->vertex(2)[0] +
                 cell->vertex(3)[0]) / 4.0,
                (cell->vertex(0)[1] +
                 cell->vertex(1)[1] +
                 cell->vertex(2)[1] +
                 cell->vertex(3)[1]) / 4.0);

        stress .push_back (sigma);

        deform .push_back (grad);

        points .push_back (midle);
    };
};

// typedef int b_t;
// typedef int c_t;
// typedef int boundary_id_t;
// typedef int material_id_t;
// template <int structdim>
// struct CellData1
// {
//     unsigned int vertices[2];
// 
//     union
//     {
//         boundary_id_t boundary_id;
//         material_id_t material_id;
//     };
// };
// struct foo
// {
//     int a[2];
//     int e;
//     union  
//     {
//         b_t b;
//         c_t c;
//     };
// };

template <uint8_t dim>
void set_without_angle (dealii::Triangulation<dim> &triangulation, 
        cdbl angle, cst n_refine)
{
    std::vector< dealii::Point< 2 > > v (8);

    cdbl x[3] = {0.0, 1.0, 2.0};
    cdbl y[3] = {0.0, 1.0, 2.0};

    v[0]  = dealii::Point<dim>(x[0], y[0]);
    v[1]  = dealii::Point<dim>(x[1], y[0]);
    v[2]  = dealii::Point<dim>(x[1], y[1]);
    v[3]  = dealii::Point<dim>(x[0], y[1]);
    v[4]  = dealii::Point<dim>(x[2], y[1]);
    v[5]  = dealii::Point<dim>(x[2], y[2]);
    v[6]  = dealii::Point<dim>(x[1], y[2]);
    v[7]  = dealii::Point<dim>(x[0], y[2]);

    std::vector< dealii::CellData< 2 > > c (3, dealii::CellData<2>());

    c[0].vertices[0] = 0; 
    c[0].vertices[1] = 1; 
    c[0].vertices[2] = 2;
    c[0].vertices[3] = 3;
    c[0].material_id = 0; 

    c[1].vertices[0] = 2; 
    c[1].vertices[1] = 4; 
    c[1].vertices[2] = 5;
    c[1].vertices[3] = 6;
    c[1].material_id = 0; 

    c[2].vertices[0] = 2; 
    c[2].vertices[1] = 6; 
    c[2].vertices[2] = 7;
    c[2].vertices[3] = 3;
    c[2].material_id = 0; 

    dealii::SubCellData b;

    b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{2, 4, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{4, 5, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{5, 6, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{6, 7, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{7, 3, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 4});

    // foo f{{1, 2}, 10,  {.b = 3}};
    // CellData1<1> g {0, 1, {.boundary_id = 0}};
    // CellData1<1> o {1, 2, {.boundary_id = 3}};
    // b.boundary_lines .push_back (dealii::CellData<1>{0, 1, {.boundary_id = 0}});
    // b.boundary_lines .push_back (dealii::CellData<1>{1, 2, {.boundary_id = 2}});
    // b.boundary_lines .push_back (dealii::CellData<1>{2, 4, {.boundary_id = 0}});
    // b.boundary_lines .push_back (dealii::CellData<1>{4, 5, {.boundary_id = 2}});
    // b.boundary_lines .push_back (dealii::CellData<1>{5, 6, {.boundary_id = 0}});
    // b.boundary_lines .push_back (dealii::CellData<1>{6, 7, {.boundary_id = 0}});
    // b.boundary_lines .push_back (dealii::CellData<1>{7, 3, {.boundary_id = 1}});
    // b.boundary_lines .push_back (dealii::CellData<1>{3, 0, {.boundary_id = 1}});

    // triangulation .create_triangulation (v, c, dealii::SubCellData());
    dealii::GridReordering<2> ::reorder_cells (c);
    triangulation .create_triangulation_compatibility (v, c, b);

    triangulation .refine_global (n_refine);
};

template <uint8_t dim>
void set_without_piece (dealii::Triangulation<dim> &triangulation, 
        cdbl angle, cst n_refine)
{
    std::vector< dealii::Point< 2 > > v (10);

    cdbl x[3] = {0.0, 1.0, 2.0};
    cdbl y[3] = {0.0, 1.0, 2.0};
    cdbl y_angle[2] = {y[1] - sin(angle) / (x[2] - x[1]),
                       y[1] + sin(angle) / (x[2] - x[1])};

    v[0]  = dealii::Point<dim>(x[0], y[0]);
    v[1]  = dealii::Point<dim>(x[1], y[0]);
    v[2]  = dealii::Point<dim>(x[2], y[0]);
    v[3]  = dealii::Point<dim>(x[2], y_angle[0]);
    v[4]  = dealii::Point<dim>(x[1], y[1]);
    v[5]  = dealii::Point<dim>(x[2], y_angle[1]);
    v[6]  = dealii::Point<dim>(x[2], y[2]);
    v[7]  = dealii::Point<dim>(x[1], y[2]);
    v[8]  = dealii::Point<dim>(x[0], y[2]);
    v[9]  = dealii::Point<dim>(x[0], y[1]);

    std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());

    c .push_back (dealii::CellData<2>{{0, 1, 4, 9}, 0});
    c .push_back (dealii::CellData<2>{{1, 2, 3, 4}, 0});
    c .push_back (dealii::CellData<2>{{4, 5, 6, 7}, 0});
    c .push_back (dealii::CellData<2>{{9, 4, 7, 8}, 0});

    dealii::SubCellData b;

    b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{3, 4, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{4, 5, 4});
    b.boundary_lines .push_back (dealii::CellData<1>{5, 6, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{6, 7, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{7, 8, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{8, 9, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{9, 0, 0});

    dealii::GridReordering<2> ::reorder_cells (c);
    triangulation .create_triangulation_compatibility (v, c, b);

    triangulation .refine_global (n_refine);
};

template <uint8_t dim>
void set_with_hole (dealii::Triangulation<dim> &triangulation, 
        cdbl angle, cst n_refine, cdbl len)
{
    cdbl x[] = {0.0, 2.0 - len, 2.0, 2.0 + len, 4.0};
    cdbl y[] = {0.0, 
                1.0 - sin(angle) / (x[2] - x[1]), 
                1.0, 
                1.0 + sin(angle) / (x[2] - x[1]),
                2.0};

    std::vector< dealii::Point< 2 > > v (16);

    v[0]   = dealii::Point<dim>(x[0], y[0]);
    v[1]   = dealii::Point<dim>(x[1], y[0]);
    v[2]   = dealii::Point<dim>(x[2], y[0]);
    v[3]   = dealii::Point<dim>(x[3], y[0]);
    v[4]   = dealii::Point<dim>(x[4], y[0]);
    v[5]   = dealii::Point<dim>(x[4], y[4]);
    v[6]   = dealii::Point<dim>(x[3], y[4]);
    v[7]   = dealii::Point<dim>(x[2], y[4]);
    v[8]   = dealii::Point<dim>(x[1], y[4]);
    v[9]   = dealii::Point<dim>(x[0], y[4]);
    v[10]  = dealii::Point<dim>(x[1], y[2]);
    v[11]  = dealii::Point<dim>(x[2], y[1]);
    v[12]  = dealii::Point<dim>(x[3], y[2]);
    v[13]  = dealii::Point<dim>(x[2], y[3]);
    v[14]  = dealii::Point<dim>(x[4], y[2]);
    v[15]  = dealii::Point<dim>(x[0], y[2]);

    std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());

    // c .push_back (dealii::CellData<2>{{ 0,  1,  8,  9}, 0});
    c .push_back (dealii::CellData<2>{{ 0,  1,  10,  15}, 0});
    c .push_back (dealii::CellData<2>{{ 15, 10,  8,  9}, 0});

    c .push_back (dealii::CellData<2>{{ 1,  2, 11, 10}, 0});
    c .push_back (dealii::CellData<2>{{ 2,  3, 12, 11}, 0});

    // c .push_back (dealii::CellData<2>{{ 3,  4,  5,  6}, 0});
    c .push_back (dealii::CellData<2>{{ 3,  4,  14,  12}, 0});
    c .push_back (dealii::CellData<2>{{ 12,  14,  5,  6}, 0});

    c .push_back (dealii::CellData<2>{{13, 12,  6,  7}, 0});
    c .push_back (dealii::CellData<2>{{10, 13,  7,  8}, 0});

    dealii::SubCellData b;

    b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{3, 4, 1});

    // b.boundary_lines .push_back (dealii::CellData<1>{4, 5, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{4, 14, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{14, 5, 0});

    b.boundary_lines .push_back (dealii::CellData<1>{5, 6, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{6, 7, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{7, 8, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{8, 9, 2});

    // b.boundary_lines .push_back (dealii::CellData<1>{9, 0, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{9, 15, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{15, 0, 0});


    b.boundary_lines .push_back (dealii::CellData<1>{10, 11, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{11, 12, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{12, 13, 4});
    b.boundary_lines .push_back (dealii::CellData<1>{13, 10, 4});

    dealii::GridReordering<2> ::reorder_cells (c);
    triangulation .create_triangulation_compatibility (v, c, b);

    triangulation .refine_global (n_refine);
};

void give_simple_sqiuare (dealii::Triangulation<2> &triangulation, 
        cst n_refine)
{
    std::vector< dealii::Point< 2 > > v (4);

    v[0]   = dealii::Point<2>(0.0, 0.0);
    v[1]   = dealii::Point<2>(1.0, 0.0);
    v[2]   = dealii::Point<2>(1.0, 1.0);
    v[3]   = dealii::Point<2>(0.0, 1.0);

    std::vector< dealii::CellData<2>> c;

    c .push_back (dealii::CellData<2>{{ 0,  1,  2,  3}, 0});

    dealii::SubCellData b;

    b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 4});
    b.boundary_lines .push_back (dealii::CellData<1>{3, 0, 1});

    dealii::GridReordering<2> ::reorder_cells (c);
    triangulation .create_triangulation_compatibility (v, c, b);

    triangulation .refine_global (n_refine);
};

void set_circ(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0, 1);

    triangulation.begin()->face(0) ->set_boundary_indicator (1);
    triangulation.begin()->face(1) ->set_boundary_indicator (2);
    triangulation.begin()->face(2) ->set_boundary_indicator (0);
    triangulation.begin()->face(3) ->set_boundary_indicator (0);

    triangulation .refine_global (n_refine);
//    {
//        dealii::Point<2> center (64.0, 64.0);
//        dealii::Triangulation<2>::active_cell_iterator
//            cell = triangulation .begin_active(),
//                 end_cell = triangulation .end();
//        for (; cell != end_cell; ++cell)
//        {
//            dealii::Point<2> midle_p(0.0, 0.0);
//
//            for (size_t i = 0; i < 4; ++i)
//            {
//                midle_p(0) += cell->vertex(i)(0);
//                midle_p(1) += cell->vertex(i)(1);
//            };
//            midle_p(0) /= 4;
//            midle_p(1) /= 4;
//
////            printf("%f %f\n", midle_p(0), midle_p(1));
//
//            if (center.distance(midle_p) < radius)
//            {
//                cell->set_material_id(1);
//            }
//            else
//                cell->set_material_id(0);
//        };
//    };
};

ElasticProblemSup<2>::BoundaryValues boundary_value (
        std::function<arr<dbl, 2>(const dealii::Point<2>)> func, cu8 id, cu8 type)
{
    return (typename ElasticProblemSup<2>::BoundaryValues{
            .function = ElasticProblemSup<2>::MyFuncFromDealii::Func(func),
            .boundari_indicator = id, 
            .type = type});
};

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
    // puts("/////");
};

enum{t_rounded_tip, t_angle_tip};
template <st type, st version> 
void give_crack(
        vec<prmt::Point<2>> &curve,
        cst num_points)
{

};

template <st type, st version> 
void give_crack(
        vec<prmt::Point<2>> &curve,
        cst num_points_on_edge,
        cst num_points_on_tip,
        cdbl weight,
        cdbl length,
        const arr<dbl, 2> center)
{

};

template <> 
void give_crack<t_rounded_tip, 1>(
        vec<prmt::Point<2>> &curve,
        cst num_points_on_edge,
        cst num_points_on_tip,
        cdbl weight,
        cdbl length,
        const arr<dbl, 2> center)
{
    cdbl PI = 3.14159265359;
    cdbl angle_step_rad = PI / num_points_on_tip;
    cdbl radius = weight / 2.0;
    for (
            dbl angle_rad = PI / 2.0; 
            abs(angle_rad - (PI / 2.0) * 3.0) > 1.e-8; 
            angle_rad += angle_step_rad
        )
    {
        dbl X = radius * sin(angle_rad) + center[0] - length / 2.0;
        dbl Y = radius * cos(angle_rad) + center[1];
        printf("%f %f\n", X, Y);
        curve .push_back (prmt::Point<2>(X, Y)); 
    };

    for (
            dbl angle_rad = 3.0 * (PI / 2.0); 
            abs(angle_rad - (2.5 * PI)) > 1.e-8; 
            angle_rad += angle_step_rad
        )
    {
        dbl X = radius * sin(angle_rad) + center[0] + length / 2.0;
        dbl Y = radius * cos(angle_rad) + center[1];
        printf("%f %f\n", X, Y);
        curve .push_back (prmt::Point<2>(X, Y)); 
    };
    // give_line_without_end_point(curve, num_points_on_edge,
    //         prmt::Point<2>(center[0], center[1] - weight / 2.0),
    //         prmt::Point<2>(0.75, 0.85));

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
    printf("type %ld\n", n_edge_on_border);
    type_edge.resize(curve.size());

    FOR(i, 0, 4)
        FOR(j, 0 + n_edge_on_border * i, n_edge_on_border + n_edge_on_border * i)
        type_edge[j] = type_border[i];
};

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<prmt::Point<2>>);

extern void set_grid(
        dealii::Triangulation< 2 >&,
        vec<prmt::Point<2>>,
        vec<prmt::Point<2>>,
        vec<st>);

void dbgputs()
{
    static st n = 0;
    printf("\x1B[31m%ld\x1B[0m \n", n);
    ++n;
};

template <int dim>
typename ElasticProblemSup<dim>::TypeCoef set_elastic_coefs (vec<std::pair<dbl, dbl>> &coef)
{
    typename ElasticProblemSup<dim>::TypeCoef elastic_coef;

    FOR(i, 0, dim) FOR(j, 0, dim) FOR(k, 0, dim) FOR(l, 0, dim)
        elastic_coef[i][j][k][l] .resize (coef.size());

    double lambda = 0.0;
    double mu     = 0.0;

    FOR(i, 0, coef.size())
    {
        lambda = coef[i].first;
        mu     = coef[i].second;

        elastic_coef[0][0][0][0][i] = lambda + 2 * mu;
        elastic_coef[1][1][1][1][i] = lambda + 2 * mu;

        elastic_coef[0][0][1][1][i] = lambda;
        elastic_coef[1][1][0][0][i] = lambda;

        elastic_coef[0][1][0][1][i] = mu;
        elastic_coef[1][0][1][0][i] = mu;
        elastic_coef[0][1][1][0][i] = mu;
        elastic_coef[1][0][0][1][i] = mu;
    };

    return elastic_coef;
};

template<int dim>
double heat_source (const dealii::Point<dim> &p)
{
//    if ((p(0) > 2.0) and (p(0) < 6.0))
//        return -6.0*p(0);
//    else
//        return -12.0*p(0);

    // return -6.0*p(0);//-6.0*p(0);//
    return 0.0;
    
    //-2 * p(0);//p(0) * (E - 2 * Nu);//-2.0*((p(0)*p(0)) + (p(1)*p(1)));
};


int main(int argc, char *argv[])
{
    cdbl pi = 3.14159265359;
    cu8 dim = 2;
    cu8 xx = 0;
    cu8 yy = 1;
    cu8 xy = 2;

    arr<prmt::Point<2>, 4> nodes;
    arr<dbl, 4> f;

    // nodes[0].x() = -1.0;
    // nodes[0].y() = -5.0;
    // f[0] = 100.0;

    // nodes[1].x() =  1.0;
    // nodes[1].y() =  -1.0;
    // f[1] = -200.0;

    // nodes[2].x() = 1.0;
    // nodes[2].y() =  1.0;
    // f[2] = 2.0;

    // nodes[3].x() = -1.0;
    // nodes[3].y() =  1.0;
    // f[3] = 3.0;

    nodes[0].x() = 0.0;
    nodes[0].y() = -3.0;
    f[0] = 100.0;

    nodes[1].x() =  5.0;
    nodes[1].y() =  0.0;
    f[1] = -100.0;

    nodes[2].x() = 0.0;
    nodes[2].y() =  1.0;
    f[2] = 2.0;

    nodes[3].x() = -1.0;
    nodes[3].y() =  0.0;
    f[3] = 3.0;

    // printf("%f %f\n", f[0], scalar_4_points_func<2> (nodes, f, nodes[0]));
    // printf("%f %f\n", f[1], scalar_4_points_func<2> (nodes, f, nodes[1]));
    // printf("%f %f\n", f[2], scalar_4_points_func<2> (nodes, f, nodes[2]));
    // printf("%f %f\n", f[3], scalar_4_points_func<2> (nodes, f, nodes[3]));

    Scalar4PointsFunc<2> func(nodes, f);
    printf("%f %f\n",f[0], func(nodes[0].x(), nodes[0].y()));
    printf("%f %f\n",f[1], func(nodes[1].x(), nodes[1].y()));
    printf("%f %f\n",f[2], func(nodes[2].x(), nodes[2].y()));
    printf("%f %f\n",f[3], func(nodes[3].x(), nodes[3].y()));

    dbl dx = 0.0001;
    printf("%f %f\n",
            (func(nodes[0].x() + dx/2., nodes[0].y()) - func(nodes[0].x() - dx/2., nodes[0].y())) / dx, 
            func.dx(nodes[0].x(), nodes[0].y()));
    printf("%f %f\n",
            (func(nodes[1].x() + dx/2., nodes[1].y()) - func(nodes[1].x() - dx/2., nodes[1].y())) / dx, 
            func.dx(nodes[1].x(), nodes[1].y()));
    printf("%f %f\n",
            (func(nodes[2].x() + dx/2., nodes[2].y()) - func(nodes[2].x() - dx/2., nodes[2].y())) / dx, 
            func.dx(nodes[2].x(), nodes[2].y()));
    printf("%f %f\n",
            (func(nodes[3].x() + dx/2., nodes[3].y()) - func(nodes[3].x() - dx/2., nodes[3].y())) / dx, 
            func.dx(nodes[3].x(), nodes[3].y()));

    printf("%f %f\n",
            (func(nodes[0].x(), nodes[0].y() + dx/2.) - func(nodes[0].x(), nodes[0].y() - dx/2.)) / dx, 
            func.dy(nodes[0].x(), nodes[0].y()));
    printf("%f %f\n",
            (func(nodes[1].x(), nodes[1].y() + dx/2.) - func(nodes[1].x(), nodes[1].y() - dx/2.)) / dx, 
            func.dy(nodes[1].x(), nodes[1].y()));
    printf("%f %f\n",
            (func(nodes[2].x(), nodes[2].y() + dx/2.) - func(nodes[2].x(), nodes[2].y() - dx/2.)) / dx, 
            func.dy(nodes[2].x(), nodes[2].y()));
    printf("%f %f\n",
            (func(nodes[3].x(), nodes[3].y() + dx/2.) - func(nodes[3].x(), nodes[3].y() - dx/2.)) / dx, 
            func.dy(nodes[3].x(), nodes[3].y()));

                    // dbgputs();
    // cdbl alpha = atof(argv[1]);

    // auto coef = set_elastic_coefs (
    //         vec<std::pair<dbl,dbl>>({std::make_pair<dbl, dbl>(1.0, 1.0)}));

    // Femenist::Function<std::array<double, dim>, dim> elastic_rhsv;

    // rhsv = source<dim>;

    // cu8 Dirichlet = ElasticProblemSup<dim>::BoundaryValues::Dirichlet;
    // cu8 Neumann   = ElasticProblemSup<dim>::BoundaryValues::Neumann;

    // cdbl max_refine = 2;

//     FOR(i, 0, 1)
//     {
//         switch (i)
//         {
//             case 0:
//                 {
//                     dbgputs();
//                     cu8 Dirichlet = 0;
//                     cu8 Neumann   = 1;
// 
//                     HeatConductionProblemSup<dim>::TypeCoef coef;
//                     coef[xx] .push_back (1.0);
//                     coef[yy] .push_back (1.0);
//                     coef[xy] .push_back (0.0);
// 
//                     Femenist::Function<double, dim> rhsv;
//                     rhsv = heat_source<dim>;
// 
//                     Femenist::MyFuncFromDealii<dim>::Func b_values =
//                         [] (const dealii::Point<dim> &p) {
//                             return p(0);};
//                             // return p(0)*p(0)*p(0);};
// 
//                     std::vector<Femenist::BoundaryValues<dim>> bound(1);
//                     bound[0].function = b_values;
//                     bound[0].boundari_indicator = 0;
//                     bound[0].type = Dirichlet;
// 
//                     // auto b_values = [] (const dealii::Point<dim> &p) {return p(0);};
//                     // vec<BoundaryValuesScalar<dim>> bound(1);
//                     // bound[0].function = b_values;
//                     // bound[0].boundari_id = 0;
//                     // bound[0].boundari_type = BVT::Dirichlet;
// 
// 
//                     dealii::Triangulation<dim> tria;
//                     dealii::GridGenerator::hyper_cube (tria, -1.0, 1.0);
//                     tria .refine_global (4);
//                     dbgputs();
// 
//                     typename dealii::Triangulation<dim>::active_face_iterator 
//                         face = tria.begin_active_face();
//                     typename dealii::Triangulation<dim>::active_face_iterator 
//                         end_face  = tria.end_face();
// 
//                     for (; face != end_face; ++face)
//                     {
//                         if (face->at_boundary())
//                         {
//                             //        if (
//                             //                ((fabs(face->vertex(0)(1) - 0.0) < 1e-10) and
//                             //                 (fabs(face->vertex(1)(1) - 0.0) < 1e-10)) or
//                             //                ((fabs(face->vertex(0)(1) - 8.0) < 1e-10) and
//                             //                 (fabs(face->vertex(1)(1) - 8.0) < 1e-10)))
//                             face ->set_boundary_indicator (0);
//                         };
//                     };
// 
//                     class ::HeatConductionProblem<dim> hc_problem (tria, coef, bound, rhsv);
//                     dbgputs();
// 
//                     REPORT hc_problem .solved ();
//                     dbgputs();
// 
//                     hc_problem .print_result (std::string("heat_condition_res_"));
//                     dbgputs();
// 
//                     break;
//                 };
// //             case 1:
// //                     FOR(j, 1, max_refine)
// //                     {
// //                     dealii::Triangulation<dim> tria;
// // 
// //                     cdbl angle = 0.0;
// // 
// //                     ::set_without_angle<2> (tria, angle, 4);
// //                     
// //                     vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};},
// //                             0,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{-1.0, 0.0};},
// //                             1,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{1.0, 0.0};},
// //                             2,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{alpha, 0.0};},
// //                             3,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{-alpha, 0.0};},
// //                             4,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);
// // 
// //                     REPORT problem .solved ();
// // 
// //                     // str name = str("angle/") + std::to_string(j) + "_";
// //                     str name = str("a") + std::to_string(j) + "_";
// // 
// //                     problem .print_result (name);
// // 
// //                     arr<dealii::Vector<dbl>, 2> stress;
// //                     arr<dealii::Vector<dbl>, 2> deform;
// //                     calc_stress_and_deform (problem, coef, stress, deform);
// //                     print_stress (problem, coef, name, stress);
// //                     };
// // 
// //                     break;
// //                 };
// //             case 1:
// //                 {
// //                     FOR(j, 1, max_refine)
// //                     {
// //                     dealii::Triangulation<dim> tria;
// // 
// //                     cdbl angle = 0.0;
// // 
// //                     ::set_without_piece<2> (tria, angle, j);
// //                     
// //                     vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             0,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -1.0};},
// //                             1,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 1.0};},
// //                             2,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, alpha * cos(angle)};},
// //                             3,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -alpha * cos(angle)};},
// //                             4,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);
// // 
// //                     REPORT problem .solved ();
// // 
// //                     str name = str("piece/") + std::to_string(j) + "_";
// // 
// //                     problem .print_result (name);
// // 
// //                     arr<dealii::Vector<dbl>, 2> stress;
// //                     arr<dealii::Vector<dbl>, 2> deform;
// //                     calc_stress_and_deform (problem, coef, stress, deform);
// //                     print_stress (problem, coef, name, stress);
// //                     };
// // 
// //                     break;
// //                 };
// //             case 2:
// //                 {
// //                     FOR(j, 1, max_refine)
// //                     {
// //                     dealii::Triangulation<dim> tria;
// // 
// //                     cdbl angle = 0.0;
// // 
// //                     ::set_with_hole<2> (tria, angle, j, 1.0);
// //                     
// //                     vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             0,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -1.0};},
// //                             1,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 1.0};},
// //                             2,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, alpha * cos(angle)};},
// //                             3,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -alpha * cos(angle)};},
// //                             4,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);
// // 
// //                     REPORT problem .solved ();
// // 
// //                     str name = str("hole/") + std::to_string(j) + "_";
// // 
// //                     problem .print_result (name);
// // 
// //                     arr<dealii::Vector<dbl>, 2> stress;
// //                     arr<dealii::Vector<dbl>, 2> deform;
// //                     calc_stress_and_deform (problem, coef, stress, deform);
// //                     print_stress (problem, coef, name, stress);
// //                     };
// // 
// //                     break;
// //                 };
// //             case 3:
// //                 {
// //                     for(dbl len = 0.1; len < 2.0; len += 0.1)
// //                     {
// //                     FOR(j, 1, max_refine)
// //                     {
// //                     dealii::Triangulation<dim> tria;
// // 
// //                     cdbl angle = 0.0;
// // 
// //                     ::set_with_hole<2> (tria, angle, j, len);
// // 
// //                     vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             0,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -1.0};},
// //                             1,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 1.0};},
// //                             2,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, alpha * cos(angle)};},
// //                             3,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha, angle] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, -alpha * cos(angle)};},
// //                             4,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);
// // 
// //                     REPORT problem .solved ();
// // 
// //                     // str name = str("angle/") + std::to_string(j) + "_";
// //                     str name = str("a") + std::to_string(j) + "_";
// // 
// //                     problem .print_result (name);
// // 
// //                     // for (size_t i = 0; i < 9; ++i)
// //                     // {
// //                     //     uint8_t im = i / (dim + 1);
// //                     //     uint8_t in = i % (dim + 1);
// // 
// //                     //     for (size_t j = 0; j < 9; ++j)
// //                     //     {
// //                     //         uint8_t jm = j / (dim + 1);
// //                     //         uint8_t jn = j % (dim + 1);
// // 
// //                     //         if (coef[im][in][jm][jn][0] > 0.0000001)
// //                     //             printf("\x1B[31m%f\x1B[0m   ", 
// //                     //                     coef[im][in][jm][jn][0]);
// //                     //         else
// //                     //             printf("%f   ", 
// //                     //                     coef[im][in][jm][jn][0]);
// //                     //     };
// //                     //     for (size_t i = 0; i < 2; ++i)
// //                     //         printf("\n");
// //                     // };
// // 
// //                     // arr<dealii::Vector<dbl>, 2> stress;
// //                     // arr<dealii::Vector<dbl>, 2> deform;
// //                     // calc_stress_and_deform (problem, coef, stress, deform);
// //                     // print_stress (problem, coef, name, stress);
// //                     // 
// //                     // dbl U = 0.0;
// //                     // for (auto i : {0,1})
// //                     //     FOR(j, 0, stress[i].size())
// //                     //         U += (stress[i][j] * deform[i][j]);
// //                     // U /= (stress[0].size() * 2);
// // 
// //                     // std::cout << "U = " << std::to_string(U) << std::endl;
// //                     // append_in_file("U.gpd",
// //                     //         std::to_string(len) + ' ' +
// //                     //         std::to_string(j)   + ' ' +
// //                     //         std::to_string(U)   + '\n');
// //                     };
// //                     // append_in_file("U.gpd", str("\n"));
// //                     };
// // 
// //                     break;
// //                 };
// //             case 4:
// //                 {
// //                     dealii::Triangulation<dim> tria;
// // 
// // //                     cdbl angle = 0.0;
// // // 
// //                     vec<prmt::Point<2>> border;
// //                     vec<st> type_border;
// //                     // give_rectangle(border, 2,
// //                     //         prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
// //                     give_rectangle_with_border_condition(
// //                             border,
// //                             type_border,
// //                             arr<st, 4>{1,3,2,4},
// //                             1,
// //                             prmt::Point<2>(0.0, 0.0), prmt::Point<2>(1.0, 1.0));
// //                     for (auto i : type_border)
// //                         printf("type %d\n", i);
// // //                         vec<prmt::LoopCondition<2>> loop_border;
// // //                         // give_rectangle_for_loop_borders(border, loop_border, 8,
// // //                         //         prmt::Point<2>(0., 0.), prmt::Point<2>(1., 1.));
// //                     vec<prmt::Point<2>> inclusion;
// //                     give_rectangle(inclusion, 1,
// //                             prmt::Point<2>(0.25, 0.25), prmt::Point<2>(0.75, 0.75));
// // //                         give_crack<t_rounded_tip, 1>(inclusion, 30);
// // // 
// //                     ::set_grid(tria, border, inclusion, type_border);
// // // 
// // //                        // ::set_tria <5> (tria, dot, material_id_for_quadrate);
// //                         tria .refine_global (2);
// // //                        set_band<2> (tria, 64.0 - 128.0 / 6.0, 64.0 + 128.0 / 6.0, 0);
// // //                        set_band<2> (tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
// //                         //                ::set_quadrate<2>(tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
// // 
// //                     // give_simple_sqiuare(tria, 2);
// //                         {
// //                         std::ofstream out ("grid-igor.eps");
// //                         dealii::GridOut grid_out;
// //                         grid_out.write_eps (tria , out);
// //                         };
// // //                         // exit(1);
// // //                         auto res = ::solved<2>(
// // //                                 tria, yung_1, puasson_1, yung_2, puasson_2,
// // //                                 loop_border); // /1.2
// //                     
// //                     // ::set_with_hole<2> (tria, angle, j);
// //                     //     {
// //                     //     std::ofstream out ("grid-igor.eps");
// //                     //     dealii::GridOut grid_out;
// //                     //     grid_out.write_eps (tria , out);
// //                     //     };
// //                     
// //                     // ::set_with_hole<2> (tria, angle, atoi(argv[2]));
// //                     vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             0,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{-1.0, 0.0};},
// //                             1,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{1.0, 0.0};},
// //                             2,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             3,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     bound .push_back (boundary_value(
// //                             [alpha] (const dealii::Point<2> &p) {
// //                             return arr<dbl, 2>{0.0, 0.0};},
// //                             4,
// //                             // Dirichlet));
// //                             Neumann));
// // 
// //                     class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);
// // 
// //                     REPORT problem .solved ();
// // 
// //                     // str name = str("angle/") + std::to_string(j) + "_";
// //                     // str name = str("a") + std::to_string(j) + "_";
// //                     str name = str("a");
// // 
// //                     problem .print_result (name);
// //                     vec<arr<arr<dbl, 2>, 2>> stress;
// //                     vec<arr<arr<dbl, 2>, 2>> deform;
// //                     vec<prmt::Point<2>> points;
// //                     calc_stress_and_deform_on_element (
// //                             problem, coef, stress, deform, points);
// //                     // print_stress (problem, coef, name, stress);
// //                     create_file("stress.gpd");
// //                     FOR (i, 0, stress.size())
// //                         append_in_file("stress.gpd", 
// //             std::to_string(points[i].x()) +
// //             " " +
// //             std::to_string(points[i].y()) +
// //             " " +
// //             std::to_string(stress[i][0][0]) +
// //             " " +
// //             std::to_string(stress[i][0][1]) +
// //             " " +
// //             std::to_string(stress[i][1][0]) +
// //             " " +
// //             std::to_string(stress[i][1][1]) +
// //             "\n"); 
// // 
// //                     // for (size_t i = 0; i < 9; ++i)
// //                     // {
// //                     //     uint8_t im = i / (dim + 1);
// //                     //     uint8_t in = i % (dim + 1);
// // 
// //                     //     for (size_t j = 0; j < 9; ++j)
// //                     //     {
// //                     //         uint8_t jm = j / (dim + 1);
// //                     //         uint8_t jn = j % (dim + 1);
// // 
// //                     //         if (coef[im][in][jm][jn][0] > 0.0000001)
// //                     //             printf("\x1B[31m%f\x1B[0m   ", 
// //                     //                     coef[im][in][jm][jn][0]);
// //                     //         else
// //                     //             printf("%f   ", 
// //                     //                     coef[im][in][jm][jn][0]);
// //                     //     };
// //                     //     for (size_t i = 0; i < 2; ++i)
// //                     //         printf("\n");
// //                     // };
// // 
// //                     // arr<dealii::Vector<dbl>, 2> stress;
// //                     // arr<dealii::Vector<dbl>, 2> deform;
// //                     // calc_stress_and_deform (problem, coef, stress, deform);
// //                     // print_stress (problem, coef, name, stress);
// //                     // 
// //                     // dbl U = 0.0;
// //                     // for (auto i : {0,1})
// //                     //     FOR(j, 0, stress[i].size())
// //                     //         U += (stress[i][j] * deform[i][j]);
// //                     // U /= (stress[0].size() * 2);
// // 
// //                     // std::cout << "U = " << std::to_string(U) << std::endl;
// //                     // append_in_file("U.gpd",
// //                     //         std::to_string(len) + ' ' +
// //                     //         std::to_string(j)   + ' ' +
// //                     //         std::to_string(U)   + '\n');
// //                     // append_in_file("U.gpd", str("\n"));
// // 
// //                     break;
// //                 };
//         };
//     };

    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};}),
    //         .boundari_indicator = 0, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -2.0};}),
    //         .boundari_indicator = 1, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 2.0};}),
    //         .boundari_indicator = 2, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
    //             // 0.0, p(1)};}),
    //             // -atof(argv[1]), 0.0};}),
    //             // 0.0, atoi(argv[1])};}),
    //             0.0, atof(argv[1]) * cos(angle)};}),
    //         .boundari_indicator = 3, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
    //             // 0.0, p(1)};}),
    //             // atof(argv[1]), 0.0};}),
    //             // 0.0, -2.0};}),
    //             0.0, -atof(argv[1]) * cos(angle)};}),
    //             // 0.0, -atof(argv[1])};}),
    //         .boundari_indicator = 4, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};}),
    //         .boundari_indicator = 0, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -2.0};}),
    //         .boundari_indicator = 1, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 2.0};}),
    //         .boundari_indicator = 2, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
    //             // 0.0, p(1)};}),
    //             // -atof(argv[1]), 0.0};}),
    //             // 0.0, atoi(argv[1])};}),
    //             0.0, atof(argv[1]) * cos(angle) * 2.0};}),
    //         .boundari_indicator = 3, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    // 
    // bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
    //         .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
    //             [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
    //             // 0.0, p(1)};}),
    //             // atof(argv[1]), 0.0};}),
    //             // 0.0, -2.0};}),
    //             0.0, -atof(argv[1]) * cos(angle) * 2.0};}),
    //             // 0.0, -atof(argv[1])};}),
    //         .boundari_indicator = 4, 
    //         // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
    //         .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    // typename ElasticProblemSup<dim>::MyFuncFromDealii::Func NeumannBoundaryValues1 =
    //     [] (const dealii::Point<dim> &p) {
    //     std::array<double, dim> res = {1.0, 0.0};//, 0.0};
    //     return res;};
    //     
    // typename ElasticProblemSup<dim>::MyFuncFromDealii::Func NeumannBoundaryValues1 =
    //     [] (const dealii::Point<dim> &p) {
    //     std::array<double, dim> res = {1.0, 0.0};//, 0.0};
    //     return res;};
    //     
    // std::vector<typename ElasticProblemSup<dim>::BoundaryValues > bound(2);
    
    //bound[0].function = NeumannBoundaryValues;
    //bound[0].boundari_indicator = 0;
    //bound[0].type = ElasticProblemSup<dim>::BoundaryValues::Neumann;
    
    //bound[1].function = NeumannBoundaryValues;
    //bound[1].boundari_indicator = 0;
    //bound[1].type = ElasticProblemSup<dim>::BoundaryValues::Neumann;

//    dealii::GridGenerator ::hyper_cube (tria, 0, 3);
//    std::vector< dealii::Point< 2 > > v (4);
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 4.0; v[1][1] = 0.0;
//    v[2][0] = 0.0; v[2][1] = 4.0;
//    v[3][0] = 4.0; v[3][1] = 4.0;
//
//    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;
//
//    tria .create_triangulation (v, c, dealii::SubCellData());

//    tria .refine_global (1);

    // class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);

    // REPORT problem .solved ();

    // problem .print_result ("output.gpd");

    // print_stress (problem, coef);

    // printf("%s %f %d\n", argv[0], atof(argv[1]), atoi(argv[2]));
    return 0;
}
/////////
