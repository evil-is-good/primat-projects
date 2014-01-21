#include <stdlib.h>
// #include <projects/deal/tests/elastic_problem/elastic_problem.h>
#include </home/primat/projects/deal/tests/elastic_problem/elastic_problem.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_reordering.h>

template<uint8_t dim>
std::array<dealii::Point<dim, double>, dim> get_grad_elastic (
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
void print_stress(const ElasticProblem<dim> &problem,
     const typename ElasticProblemSup<dim>::TypeCoef &C)
{
    enum {x, y, z};
    char suffix[3] = {'x', 'y', 'z'};
    
    // vec<dealii::Point<2>> stress(problem.system_equations.x.size());
    arr<dealii::Vector<dbl>, 2> stress;
    stress[0].reinit(problem.system_equations.x.size());
    stress[1].reinit(problem.system_equations.x.size());
    vec<u8> divider(problem.system_equations.x.size());

    typename dealii::DoFHandler<2>::active_cell_iterator cell =
        problem.domain.dof_handler.begin_active();

    typename dealii::DoFHandler<2>::active_cell_iterator endc =
        problem.domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        FOR(i, 0, 4)
        {
            auto deform = 
                ::get_grad_elastic<dim> (cell, problem.system_equations.x, i);  

            cdbl sigma_xx = 
                C[x][x][x][x][0] * deform[x](x) +
                C[x][x][x][y][0] * deform[x](y) +
                C[x][x][y][x][0] * deform[y](x) +
                C[x][x][y][y][0] * deform[y](y);

            cdbl sigma_xy = 
                C[x][y][x][x][0] * deform[x](x) +
                C[x][y][x][y][0] * deform[x](y) +
                C[x][y][y][x][0] * deform[y](x) +
                C[x][y][y][y][0] * deform[y](y);

            cdbl sigma_yx = 
                C[y][x][x][x][0] * deform[x](x) +
                C[y][x][x][y][0] * deform[x](y) +
                C[y][x][y][x][0] * deform[y](x) +
                C[y][x][y][y][0] * deform[y](y);

            cdbl sigma_yy = 
                C[y][y][x][x][0] * deform[x](x) +
                C[y][y][x][y][0] * deform[x](y) +
                C[y][y][y][x][0] * deform[y](x) +
                C[y][y][y][y][0] * deform[y](y);

            // stress[cell->vertex_dof_index(i,0)] += 
            //     // dealii::Point<2>(deform[x](x), deform[x][y]);
            //     dealii::Point<2>(sigma_xx, sigma_xy);

            // stress[cell->vertex_dof_index(i,1)] += 
            //     // dealii::Point<2>(deform[y](x), deform[y][y]);
            //     dealii::Point<2>(sigma_yx, sigma_yy);

            stress[x][cell->vertex_dof_index(i, x)] += sigma_xx;
            stress[x][cell->vertex_dof_index(i, y)] += sigma_xy;
            stress[y][cell->vertex_dof_index(i, x)] += sigma_yx;
            stress[y][cell->vertex_dof_index(i, y)] += sigma_yy;

            divider[cell->vertex_dof_index(i,0)] += 1;
            divider[cell->vertex_dof_index(i,1)] += 1;
        };
    };

    FOR(i, 0, divider.size())
    {
        stress[x][i] /= divider[i];
        stress[y][i] /= divider[i];
    };

    for (auto i : {x,y})
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (problem.domain.dof_handler);

        data_out.add_data_vector (stress[i], "x y");
        data_out.build_patches ();

        std::string file_name = str("stress_") + suffix[i] + ".gpd";

        std::ofstream output (file_name.data());
        data_out.write_gnuplot (output);
    };
};

typedef int b_t;
typedef int c_t;
typedef int boundary_id_t;
typedef int material_id_t;
template <int structdim>
struct CellData1
{
    unsigned int vertices[2];

    union
    {
        boundary_id_t boundary_id;
        material_id_t material_id;
    };
};
struct foo
{
    int a[2];
    int e;
    union  
    {
        b_t b;
        c_t c;
    };
};

template <uint8_t dim>
void set_without_angle (dealii::Triangulation<dim> &triangulation, 
        cdbl angle, cst n_refine)
{
    std::vector< dealii::Point< 2 > > v (8);

    cdbl x[3] = {0.0, 2.0, 3.0};
    cdbl y[3] = {0.0, 1.0, 3.0};

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
        cdbl angle, cst n_refine)
{
    cdbl x[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    cdbl y[] = {0.0, 
                1.0 - sin(angle) / (x[2] - x[1]), 
                1.0, 
                1.0 + sin(angle) / (x[2] - x[1]),
                2.0};

    std::vector< dealii::Point< 2 > > v (14);

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

    std::vector< dealii::CellData<2>> c; //(3, dealii::CellData<2>());

    c .push_back (dealii::CellData<2>{{ 0,  1,  8,  9}, 0});
    c .push_back (dealii::CellData<2>{{ 1,  2, 11, 10}, 0});
    c .push_back (dealii::CellData<2>{{ 2,  3, 12, 11}, 0});
    c .push_back (dealii::CellData<2>{{ 3,  4,  5,  6}, 0});
    c .push_back (dealii::CellData<2>{{13, 12,  6,  7}, 0});
    c .push_back (dealii::CellData<2>{{10, 13,  7,  8}, 0});

    dealii::SubCellData b;

    b.boundary_lines .push_back (dealii::CellData<1>{0, 1, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{1, 2, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{2, 3, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{3, 4, 1});
    b.boundary_lines .push_back (dealii::CellData<1>{4, 5, 0});
    b.boundary_lines .push_back (dealii::CellData<1>{5, 6, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{6, 7, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{7, 8, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{8, 9, 2});
    b.boundary_lines .push_back (dealii::CellData<1>{9, 0, 0});

    b.boundary_lines .push_back (dealii::CellData<1>{10, 11, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{11, 12, 3});
    b.boundary_lines .push_back (dealii::CellData<1>{12, 13, 4});
    b.boundary_lines .push_back (dealii::CellData<1>{13, 10, 4});

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

int main(int argc, char *argv[])
{
    cdbl pi = 3.14159265359;
    // str s = str("sdf") + str("sd") + "sdfswdfw"_s;
    str s = str("asda") + "sdsf"_s;
    const uint8_t dim = 2;

//    std::array<Femenist::Function<double, dim>, 3 > coef;
    ElasticProblemSup<dim>::TypeCoef coef;
    Femenist::Function<std::array<double, dim>, dim> rhsv;
//    Femenist::Function<std::array<double, dim>, dim> bound;

    double lambda = 0.0;
    double mu     = 0.0;

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    coef[i][j][k][l] .resize (2);
    
    lambda = 1.0;
    mu     = 1.0;

    coef[0][0][0][0][0] = lambda + 2 * mu;
    coef[1][1][1][1][0] = lambda + 2 * mu;

    coef[0][0][1][1][0] = lambda;
    coef[1][1][0][0][0] = lambda;

    coef[0][1][0][1][0] = mu;
    coef[1][0][1][0][0] = mu;
    coef[0][1][1][0][0] = mu;
    coef[1][0][0][1][0] = mu;
    
    lambda = 1.0;
    mu     = 1.0;

    coef[0][0][0][0][1] = lambda + 2 * mu;
    coef[1][1][1][1][1] = lambda + 2 * mu;

    coef[0][0][1][1][1] = lambda;
    coef[1][1][0][0][1] = lambda;

    coef[0][1][0][1][1] = mu;
    coef[1][0][1][0][1] = mu;
    coef[0][1][1][0][1] = mu;
    coef[1][0][0][1][1] = mu;
    
    rhsv    = source<dim>;//const0<dim>;//////
//    bound   = boundary<dim>;

    dealii::Triangulation<dim> tria;

//    const size_t material_id_for_quadrate[4][4] =
//    {
//        {0, 0, 0, 0},
//        {0, 1, 1, 0},
//        {0, 1, 1, 0},
//        {0, 0, 0, 0}
//    };
//
//    const double dot[5] = 
//    {
//        (0.0),
//        (64.0 - i / 2.0),
//        (64.0),
//        (64.0 + i / 2.0),
//        (128.0)
//    };
//
//    ::set_tria <5> (tria, dot, material_id_for_quadrate);

    // ::set_circ(tria, 0, 4);
    // ::set_without_angle<2> (tria, 1.0, atoi(argv[2]));

    // cdbl angle = pi / 6.0; //0.0; //pi / 100.0;
    cdbl angle = 0.0;

    // ::set_without_piece<2> (tria, angle, atoi(argv[2]));
    ::set_with_hole<2> (tria, angle, atoi(argv[2]));

    typename ElasticProblemSup<dim>::MyFuncFromDealii::Func NeumannBoundaryValues =
        [] (const dealii::Point<dim> &p) {
        std::array<double, dim> res = {1.0, 0.0};//, 0.0};
        return res;};

    typename ElasticProblemSup<dim>::MyFuncFromDealii::Func DirichletBoundaryValues =
        [] (const dealii::Point<dim> &p) {
        std::array<double, dim> res = {1.0, 0.0};//, p(0)};
        return res;};

    vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
    
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
    
    bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
            .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
                [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};}),
            .boundari_indicator = 0, 
            // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
            .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
            .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
                [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, -2.0};}),
            .boundari_indicator = 1, 
            // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
            .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
            .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
                [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 2.0};}),
            .boundari_indicator = 2, 
            // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
            .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
            .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
                [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
                // 0.0, p(1)};}),
                // -atof(argv[1]), 0.0};}),
                // 0.0, atoi(argv[1])};}),
                0.0, atof(argv[1]) * cos(angle) * 2.0};}),
            .boundari_indicator = 3, 
            // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
            .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
    bound .push_back (typename ElasticProblemSup<dim>::BoundaryValues{
            .function = ElasticProblemSup<dim>::MyFuncFromDealii::Func(
                [argv, angle] (const dealii::Point<2> &p) {return arr<dbl, 2>{
                // 0.0, p(1)};}),
                // atof(argv[1]), 0.0};}),
                // 0.0, -2.0};}),
                0.0, -atof(argv[1]) * cos(angle) * 2.0};}),
                // 0.0, -atof(argv[1])};}),
            .boundari_indicator = 4, 
            // .type = ElasticProblemSup<dim>::BoundaryValues::Dirichlet});
            .type = ElasticProblemSup<dim>::BoundaryValues::Neumann});
    
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

    class ::ElasticProblem<dim> problem (tria, coef, bound, rhsv);

    REPORT problem .solved ();

    problem .print_result ("output.gpd");

    print_stress (problem, coef);

    printf("%s %f %d\n", argv[0], atof(argv[1]), atoi(argv[2]));
    return 0;
}
/////////
