//#include "../../elastic_test_on_cell/sources/cgal/cgal.h"
//
//#include <stdlib.h>
//#include <deal.II/grid/grid_generator.h>
//#include <deal.II/grid/tria_accessor.h>
//#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/grid_out.h>
//#include <deal.II/grid/grid_reordering.h>
//#include <ctime>
#include "./head.h"
#include <projects/deal/tests/heat_conduction_problem_on_cell/heat_conduction_problem_on_cell.h>
#include "../../../calculation_core/src/blocks/general/4_points_function/4_points_function.h"

// typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
// typedef CGAL::Triangulation_vertex_base_2<K> Vb;
// typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
// typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
// typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
// typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
// 
// typedef CDT::Vertex_handle Vertex_handle;
// typedef CDT::Point Point;

//#define FOR_I(begin, end) for(size_t i = begin; i < end; ++i)
//#define FOR_J(begin, end) for(size_t j = begin; j < end; ++j)

const size_t PI = 3.14159265359;

class Hexagon
{
    public:
        Hexagon(): center(0.0, 0.0), radius(0.0) {};
        Hexagon(const dealii::Point<2> &c, const double r)
        {
            center = c;
            radius = r;

            point[0] = dealii::Point<2>(center(0), center(1) - radius);
            point[1] = dealii::Point<2>(center(0) + radius * (sqrt(3.0) / 2.0),
                                             center(1) - radius / 2.0);
            point[2] = dealii::Point<2>(center(0) + radius * (sqrt(3.0) / 2.0), 
                                             center(1) + radius / 2.0);
            point[3] = dealii::Point<2>(center(0), center(1) + radius);
            point[4] = dealii::Point<2>(center(0) - radius * (sqrt(3.0) / 2.0), 
                                             center(1) + radius / 2.0);
            point[5] = dealii::Point<2>(center(0) - radius * (sqrt(3.0) / 2.0),
                                             center(1) - radius / 2.0);
        };

        bool include (const dealii::Point<2> &p)
        {
            bool res = false;

            if (radius < 1e-10)
                return res;

//            printf("%f %f %f %f %f %f\n", p1(0), p2(0), p3(0), p4(0), p5(0), p6(0));
//            printf("%f %f %f %f %f %f\n", p1(1), p2(1), p3(1), p4(1), p5(1), p6(1));

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
                above_the_line(point[0], point[1]) +
                above_the_line(point[1], point[2]) +
                above_the_line(point[2], point[3]) +
                above_the_line(point[3], point[4]) +
                above_the_line(point[4], point[5]) +
                above_the_line(point[5], point[0]);

//            printf("%f %f %f %f %f %f\n", 
//                above_the_line(p1, p2),
//                above_the_line(p2, p3),
//                above_the_line(p3, p4),
//                above_the_line(p5, p6),
//                above_the_line(p6, p1));

            if (sum_positive_sign == 6)
                res = true;

            return res;
        };

    std::array<dealii::Point<2>, 6> point;

    private:
        dealii::Point<2> center;
        double radius;
};

void tri_import(dealii::Triangulation<2> &tri, const char *filename)
{
    FILE *F;
    
    F = fopen(filename,"r");
    if (F) printf("12345\n");

    double aa,ba,ca,da;
    
    fscanf(F,"%lf %lf %lf %lf",&aa,&ba,&ca,&da);
    
    int num_v;
    fscanf(F,"%d",&num_v);
    std::vector< dealii::Point< 2 > > v (num_v);
    for (unsigned int i=0; i<num_v; ++i)
    {
        fscanf(F,"%lf %lf",&v[i][0],&v[i][1]);
    };
    
    int num_c;
    fscanf(F,"%d",&num_c);
    std::vector< dealii::CellData< 2 > > c (num_c, dealii::CellData<2>());;
    for (unsigned int i=0; i<num_c; ++i)
    {
        unsigned int m_id = 0; 
        fscanf(F,"%d %d %d %d %d",&c[i].vertices[0],
                                   &c[i].vertices[1],
                                   &c[i].vertices[2],
                                   &c[i].vertices[3],
                                   &m_id);
        c[i].material_id = m_id;
    };
    
    tri.create_triangulation (v, c, dealii::SubCellData());
};

template<int dim>
double const0 (const dealii::Point<dim> &p)
{
    return 0.0;
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

    double f1 = index_vertex == 0 ? 1.0 : 0.0; //solution(cell->vertex_dof_index (0, 0));
    double f2 = index_vertex == 1 ? 1.0 : 0.0; //solution(cell->vertex_dof_index (1, 0));
    double f3 = index_vertex == 2 ? 1.0 : 0.0; //solution(cell->vertex_dof_index (2, 0));
    double f4 = index_vertex == 3 ? 1.0 : 0.0; //solution(cell->vertex_dof_index (3, 0));

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    
    if ((cell->vertex_dof_index(1,0) == 35) and (index_vertex == 1))
    {
        printf("!!!!!!i %f %f %f %f %f\n", b, c, d, 
        (x3*y1*f2-x4*y1*f2+x4*y3*f2-x1*y3*f2+x1*y4*f2-x3*y4*f2)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1),
        (x3*y1*f2-x4*y1*f2+x4*y3*f2-x1*y3*f2+x1*y4*f2-x3*y4*f2));
        printf("!!!!!!i %f %f %f %f %f %f %f %f\n", x1, x2, x3, x4, y1, y2, y3, y4);
        // printf("!!!!!!!! %d %d %d %d\n", 
        //         cell->vertex_dof_index(0,0),
        //         cell->vertex_dof_index(1,0),
        //         cell->vertex_dof_index(2,0),
        //         cell->vertex_dof_index(3,0));
    };


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

void set_circ(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
//    puts("1111111111111111111111111111111111111111111");
    dealii::GridGenerator ::hyper_cube (triangulation, 0, 128);
    triangulation .refine_global (n_refine);
    {
        dealii::Point<2> center (64.0, 64.0);
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
            midle_p(0) /= 4;
            midle_p(1) /= 4;

//            printf("%f %f\n", midle_p(0), midle_p(1));

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


double get_hashin(const double c, const double coef_1, const double coef_2)
{
    return coef_1 * (1.0 + c / (coef_1 / (coef_2 - coef_1) + (1.0 - c) / 2.0));
};

double get_vanin(const double c, const double coef_1, const double coef_2,
                 const double alpha, const size_t n)
{
    const double l1 = coef_1;
    const double l2 = coef_2;

    double l0 = l1 * 
        (1.0 + c + (1.0 - c) * l1 / l2) / 
        (1.0 - c + (1.0 + c) * l1 / l2);

    double a1 = n * n * (n - 1) * l0 / l1;
    double a2 = pow((1.0 - l2 / l1) / 
//            (1.0 + l2 / l1), 2.0);
    (1.0 - c + (1.0 + c) * l2 / l1), 2.0);
    double a3 = pow(sin(alpha), 1 * 2.0) / pow(PI, n);
    double a4 = pow(c, 2.0) - pow(c, 2.0 * n) * 
        (pow((1.0 - l2 / l1) / (1.0 + l2 / l1), 2.0));
    return l0 * (1.0 + a1 * a2 * a3 * a4);
};

template <size_t num_points>
void set_hexagon_brave(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{

    const size_t num_cells = num_points - 1;
    const size_t height = len_edge * (3.0 / 4.0);//(sqrt(3.0) / 2.0);

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * 2.0 * height) / num_points, 
                    (i * 1.0 * len_edge) / num_points);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

        dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            c[cell_number].material_id = 0;

            {
                dealii::Point<2> center (height, 0.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (0.0, len_edge / 2.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (2.0 * height, len_edge / 2.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (height, len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

template <size_t num_points>
void set_hexagon_grid(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{

    const size_t num_cells = num_points - 1;
    const double height = len_edge * (sqrt(3.0) / 2.0);
//    printf("%f %f\n", `<args>`);

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * 2.0 * height) / num_points, 
                    (i * 3.0 * len_edge) / num_points);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

        dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            c[cell_number].material_id = 0;

           {
               dealii::Point<2> center (height, 0.0);
               if (Hexagon(center, radius) .include(midle_p))
                   c[cell_number].material_id = 1;
           }

//            {
//                dealii::Point<2> center (3.0 * height, 0.0);
//                if (Hexagon(center, radius) .include(midle_p))
//                    c[cell_number].material_id = 1;
//            }

           {
               dealii::Point<2> center (0.0, 1.5 * len_edge);
               if (Hexagon(center, radius) .include(midle_p))
                   c[cell_number].material_id = 1;
           }

           {
               dealii::Point<2> center (2.0 * height, 1.5 * len_edge);
               if (Hexagon(center, radius) .include(midle_p))
                   c[cell_number].material_id = 1;
           }

//            {
//                dealii::Point<2> center (4.0 * height, 1.5 * len_edge);
//                if (Hexagon(center, radius) .include(midle_p))
//                    c[cell_number].material_id = 1;
//            }

           {
               dealii::Point<2> center (height, 3.0 * len_edge);
               if (Hexagon(center, radius) .include(midle_p))
                   c[cell_number].material_id = 1;
           }


            // {
            //     dealii::Point<2> center (height, 0.0);
            //     if (center.distance(midle_p) < radius)
            //         c[cell_number].material_id = 1;
            // }

            // {
            //     dealii::Point<2> center (0.0, 1.5 * len_edge);
            //     if (center.distance(midle_p) < radius)
            //         c[cell_number].material_id = 1;
            // }

            // {
            //     dealii::Point<2> center (2.0 * height, 1.5 * len_edge);
            //     if (center.distance(midle_p) < radius)
            //         c[cell_number].material_id = 1;
            // }

            // {
            //     dealii::Point<2> center (height, 3.0 * len_edge);
            //     if (center.distance(midle_p) < radius)
            //         c[cell_number].material_id = 1;
            // }

        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

void set_hexagon_grid_pure(dealii::Triangulation< 2 > &triangulation, 
        cdbl total_area,
        cdbl share_includ)
        // const double len_edge,
        // const double radius)
{
    // double Ro = len_edge;
    // double ro = radius;
    // double Ri = Ro * (sqrt(3.0) / 2.0);
    // double ri = ro * (sqrt(3.0) / 2.0);

    // dbl Ro = 4.0;
    // dbl Ri = Ro * (sqrt(3.0) / 2.0);
    // // dbl ri = Ri - 0.025;
    // dbl ro = 3.9613;//Ro - 0.05;//ri / (sqrt(3.0) / 2.0);
    // dbl ri = ro * (sqrt(3.0) / 2.0);

    cdbl Ro = sqrt(total_area / (3.0 * sqrt(3.0)));
    cdbl Ri = Ro * (sqrt(3.0) / 2.0);
    cdbl ro = sqrt((total_area * share_includ) / (3.0 * sqrt(3.0)));
    cdbl ri = ro * (sqrt(3.0) / 2.0);

    printf("Radius Ro=%f Ri=%f ro=%f ri=%f\n", Ro, Ri, ro, ri);

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

template<int dim>
double square (const dealii::Point<dim> &p)
{
    uint8_t num_true = 0;
    for (size_t i = 0; i < dim; ++i)
    if ((p[i] > 1.0) && (p[i] < 3.0))
        ++num_true;

    if (num_true == dim)
        return 2.0;
    else
        return 1.0;
};

template<int dim>
double band (const dealii::Point<dim> &p)
{
    if ((p[0] > 1.0) && (p[0] < 3.0))
        return 2.0;
    else
        return 1.0;
};
template <size_t num_points>
void set_line(dealii::Triangulation< 2 > &triangulation, 
        const double hx,
        const double hy,
        const double deriver)
{
    const size_t num_cells = num_points - 1;

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * hx) / (num_points - 1), 
                    (i * hy) / (num_points - 1));
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

            dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            if (midle_p(0) < deriver)
                c[cell_number].material_id = 1;
            else
                c[cell_number].material_id = 0;
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};


// void set_grid(dealii::Triangulation< 2 > &triangulation)
// {
//     std::vector<dealii::Point<2> > v;
//     std::vector<dealii::CellData<2> > c;
// 
//     CDT cdt;
//     std::vector<Vertex_handle> vec_of_vertices;
//     std::vector<std::vector<Vertex_handle> > vec_of_domains;
// 
// //    double x_begin = -1.0;
// //    double x_end = 1.0;
// //    double dx = 0.05;
// //    double x_current;
// //    size_t num_node_x = (x_end - x_begin) / dx + 1;
// //
// //    double mult = 2.0;
// //    double shift = 4.0;
// //
// //    for(size_t item_node_x = 0; item_node_x < num_node_x; ++item_node_x)
// //    {
// //         x_current = x_begin + item_node_x * dx;
// //         vec_of_vertices.push_back( cdt.insert( 
// //                     Point(x_current*mult+shift, 
// //                         sqrt(1.0 - x_current * x_current)*mult+shift )) );  
// //    }
// //
// //    for(size_t item_node_x = 1; item_node_x < num_node_x - 1; ++item_node_x)
// //    {
// //        x_current = x_end - item_node_x * dx;
// //        vec_of_vertices.push_back( cdt.insert( 
// //                    Point(x_current*mult+shift, 
// //                        - sqrt(1.0 - x_current * x_current)*mult+shift )) );  
// //    }
// //    
//     Hexagon hexagon(dealii::Point<2>(0.5, 0.5), 0.5/3.141592653589);
// 
//    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0,  0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0,  1.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0,  1.0) ));
// 
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[0](0), hexagon.point[0](1))));
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[1](0), hexagon.point[1](1))));
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[2](0), hexagon.point[2](1))));
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[3](0), hexagon.point[3](1))));
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[4](0), hexagon.point[4](1))));
// //    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[5](0), hexagon.point[5](1))));
// 
//     // vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 1.0/3.0) ));
//     // vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 1.0/3.0) ));
//     // vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 2.0/3.0) ));
//     // vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 2.0/3.0) ));
// 
// 
//     vec_of_domains.push_back(vec_of_vertices);
//     vec_of_vertices.clear();     
//     
// //    vec_of_vertices.push_back(cdt.insert(Point(0.0, 0.0) ));
// //    vec_of_vertices.push_back(cdt.insert(Point(128.0, 0.0) ));
// //    vec_of_vertices.push_back(cdt.insert(Point(128.0, 128.0) ));
// //    vec_of_vertices.push_back(cdt.insert(Point(0.0, 128.0) ));
// //
// //    vec_of_domains.push_back(vec_of_vertices);
// //    vec_of_vertices.clear();  
//     
//   double width_outer_domain = 1.0;
//   double height_outer_domain = 1.0;
//   
//   double begin_x_outer_domain = 0.0;
//   double begin_y_outer_domain = 0.0;
//   
//   size_t num_nodes_by_x = 0;
//   size_t num_nodes_by_y = 0;
//   
//   double end_x_outer_domain = begin_x_outer_domain + width_outer_domain;
//   double end_y_outer_domain = begin_y_outer_domain + height_outer_domain;
//   
// 
//   
//   double step_by_x = width_outer_domain / (num_nodes_by_x + 1); 
//   double step_by_y = height_outer_domain / (num_nodes_by_y + 1); 
//   
//   for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
//   {
//      vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain + item_node * step_by_x,
//                                                  begin_y_outer_domain ) ));      
//   }
//   
//   for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
//   {
//      vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain,
//                                                  begin_y_outer_domain +  item_node * step_by_y) ));      
//   }
//   
//   for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
//   {
//      vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain - item_node * step_by_x,
//                                                  end_y_outer_domain ) ));      
//   }
//   
//   for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
//   {
//      vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain,
//                                                  end_y_outer_domain -  item_node * step_by_y) ));      
//   }
//   
//     vec_of_domains.push_back(vec_of_vertices);
//     vec_of_vertices.clear(); 
//     
//     for(auto it = vec_of_domains.begin(); it != vec_of_domains.end(); ++it) 
//     {
//        for(auto vit = it->begin() + 1; vit != it->end(); ++vit)
//        {
//           cdt.insert_constraint( *(vit - 1), *vit );
//        }
//        cdt.insert_constraint( *( it->end() - 1), *( it->begin() ) );
//     }
//     
//     
//     std::list<Point> list_of_seeds;
//     list_of_seeds.push_back(Point(0.5, 0.5));
// 
//     std::cout << "Meshing the domain..." << std::endl;
//   
//     CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
//                                Criteria(), true);
//     CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
//                                Criteria());
// //    CGAL::refine_Delaunay_mesh_2(cdt, Criteria());
// 
//     std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
//     std::cout << "Number of finite faces: " << cdt.number_of_faces() << std::endl;
//     
//     size_t mesh_faces_counter = 0;
//   
//     for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) 
//     {
//         if(fit->is_in_domain()) 
//             ++mesh_faces_counter;
//     }
//     
//     std::cout << "Number of faces in the mesh domain: " << mesh_faces_counter << std::endl;
//     
//     CDT::Triangle trg;
//     
//     size_t mat_id; 
// 
//     for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
//     {
// //        puts("!!!!!!!!!!!!!!");
//         if(fit->is_in_domain())
//         { 
//             mat_id = 0;
//         }
//         else
//         {
//             mat_id = 1;
//         }      
//       
//         trg = cdt.triangle(fit);
//       
//         size_t t = v.size()*10;
//         size_t indx[7] = {t+0, t+1, t+2, t+3, t+4, t+5, t+6};
//       
//         double middle_segment01_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()));
//         double middle_segment01_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()));
//       
//         double middle_segment12_x = 0.5 * (CGAL::to_double(trg[1].x()) + CGAL::to_double(trg[2].x()));
//         double middle_segment12_y = 0.5 * (CGAL::to_double(trg[1].y()) + CGAL::to_double(trg[2].y()));
// 
//         double middle_segment02_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[2].x()));
//         double middle_segment02_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[2].y()));
//       
//         double center_of_mass_x = (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()) +
//                                    CGAL::to_double(trg[2].x())) / 3.0;
//         double center_of_mass_y = (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()) +
//                                    CGAL::to_double(trg[2].y())) / 3.0;
//       
//         FOR_I(0, v.size())
//         {
//             if (
//                     (fabs(v[i](0) - trg[0].x()) < 1e-10) and
//                     (fabs(v[i](1) - trg[0].y()) < 1e-10))
//                 indx[0] = i;
//             if (
//                     (fabs(v[i](0) - trg[1].x()) < 1e-10) and
//                     (fabs(v[i](1) - trg[1].y()) < 1e-10))
//                 indx[1] = i;
//             if (
//                     (fabs(v[i](0) - trg[2].x()) < 1e-10) and
//                     (fabs(v[i](1) - trg[2].y()) < 1e-10))
//                 indx[2] = i;
//             if (
//                     (fabs(v[i](0) - middle_segment01_x) < 1e-10) and
//                     (fabs(v[i](1) - middle_segment01_y) < 1e-10))
//                 indx[3] = i;
//             if (
//                     (fabs(v[i](0) - middle_segment12_x) < 1e-10) and
//                     (fabs(v[i](1) - middle_segment12_y) < 1e-10))
//                 indx[4] = i;
//             if (
//                     (fabs(v[i](0) - middle_segment02_x) < 1e-10) and
//                     (fabs(v[i](1) - middle_segment02_y) < 1e-10))
//                 indx[5] = i;
//             if (
//                     (fabs(v[i](0) - center_of_mass_x) < 1e-10) and
//                     (fabs(v[i](1) - center_of_mass_y) < 1e-10))
//                 indx[6] = i;
//         };
// 
//         if (indx[0] == t)
//         {
//             v.push_back(dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
//             indx[0] = v.size()-1;
//         }
//         if (indx[1] == t+1)
//         {
//             v.push_back(dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
//             indx[1] = v.size()-1;
//         }
//         if (indx[2] == t+2)
//         {
//             v.push_back(dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
//             indx[2] = v.size()-1;
//         }
//         if (indx[3] == t+3)
//         {
//             v.push_back(
// //                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
// //                    dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
//                     dealii::Point<2>( middle_segment01_x, middle_segment01_y ));
//             indx[3] = v.size()-1;
//         }
//         if (indx[4] == t+4)
//         {
//             v.push_back(
// //                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
// //                    dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
//                     dealii::Point<2>( middle_segment12_x, middle_segment12_y ));
//             indx[4] = v.size()-1;
//         }
//         if (indx[5] == t+5)
//         {
//             v.push_back(
// //                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
// //                    dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
//                     dealii::Point<2>( middle_segment02_x, middle_segment02_y ));
//             indx[5] = v.size()-1;
//         }
//         if (indx[6] == t+6)
//         {
//             v.push_back(dealii::Point<2>( center_of_mass_x, center_of_mass_y));
//             indx[6] = v.size()-1;
//         }
// 
// //        printf("%d %d %d %d %d %d %d\n", 
// //                indx[0],
// //                indx[1],
// //                indx[2],
// //                indx[3],
// //                indx[4],
// //                indx[5],
// //                indx[6]);
// 
// //        v.push_back(dealii::Point<2>(0.0, 0.0));
// //        v.push_back(dealii::Point<2>(1.0, 0.0));
// //        v.push_back(dealii::Point<2>(2.0, 0.0));
// //        v.push_back(dealii::Point<2>(0.0, 2.0));
// //        v.push_back(dealii::Point<2>(1.0, 2.0));
// //        v.push_back(dealii::Point<2>(2.0, 2.0));
// 
//         {
//             dealii::CellData<2> temp_c;
// 
//             temp_c.vertices[0] = indx[0];
//             temp_c.vertices[1] = indx[3];
//             temp_c.vertices[2] = indx[6];
//             temp_c.vertices[3] = indx[5];
//             temp_c.material_id = mat_id;
// 
//             c.push_back(temp_c);
//         }
// 
//         {
//             dealii::CellData<2> temp_c;
// 
//             temp_c.vertices[0] = indx[1];
//             temp_c.vertices[1] = indx[4];
//             temp_c.vertices[2] = indx[6];
//             temp_c.vertices[3] = indx[3];
//             temp_c.material_id = mat_id;
// 
//             c.push_back(temp_c);
//         }
// 
//         {
//             dealii::CellData<2> temp_c;
// 
//             temp_c.vertices[0] = indx[2];
//             temp_c.vertices[1] = indx[5];
//             temp_c.vertices[2] = indx[6];
//             temp_c.vertices[3] = indx[4];
//             temp_c.material_id = mat_id;
// 
//             c.push_back(temp_c);
//         }
// 
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[0];
// //            temp_c.vertices[1] = indx[3];
// //            temp_c.vertices[2] = indx[1];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// //
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[1];
// //            temp_c.vertices[1] = indx[4];
// //            temp_c.vertices[2] = indx[2];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// //
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[2];
// //            temp_c.vertices[1] = indx[5];
// //            temp_c.vertices[2] = indx[0];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// 
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[0];
// //            temp_c.vertices[1] = indx[1];
// //            temp_c.vertices[2] = indx[3];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// //
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[1];
// //            temp_c.vertices[1] = indx[2];
// //            temp_c.vertices[2] = indx[4];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// //
// //        {
// //            dealii::CellData<2> temp_c;
// //
// //            temp_c.vertices[0] = indx[2];
// //            temp_c.vertices[1] = indx[0];
// //            temp_c.vertices[2] = indx[5];
// //            temp_c.vertices[3] = indx[6];
// //            temp_c.material_id = mat_id;
// //
// //            c.push_back(temp_c);
// //        }
// 
//     }
// 
//     printf("%d %d\n", v.size(), c.size());
//     puts("1");
// //    dealii::GridReordering<2>::invert_all_cells_of_negative_grid(v, c);
//     dealii::GridReordering<2>::reorder_cells(c);
//     puts("2");
//     triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
//     puts("3");
// 
// };

template<size_t dim>
std::array<double, 3> solved (dealii::Triangulation<dim> &triangulation,
        const double coef_1, const double coef_2)
{
    std::array<std::vector<double>, 3> coef;

    coef[0] .resize (2);
    coef[1] .resize (2);
    coef[2] .resize (2);

    coef[0][0] = coef_1; coef[0][1] = coef_2;
    coef[1][0] = coef_1; coef[1][1] = coef_2;
    coef[2][0] = 0.0;    coef[2][1] = 0.0;

    class ::HeatConductionProblemOnCell<dim> problem (triangulation, coef);

    REPORT problem .solved ();

    problem .print_result ("res_");

    dbl max_s = 0.0;
    for (auto i : problem.solution[0])
        if (i > max_s)
            max_s = i;

    // dealii::Vector<dealii::Point<2>> coor(problem.solution[0].size());
    // auto cell = dof_handler.begin_active();
    // auto endc = dof_handler.end();
    // for (; cell != endc; ++cell)
    // {
    //     for (st i = 0; i < 4; ++i)
    //     {
    //         coor[cell->vertex_dof_index(i,0)] = cell->vertex(i);
    //     };
    // };

    // {
    //     dbl integ = 0.0;
    //
    //     {
    //         dealii::QGauss<dim>  quadrature_formula(2);
    //
    //         dealii::FEValues<dim> fe_values (dof_handler.get_fe(), quadrature_formula,
    //                 dealii::update_quadrature_points | dealii::update_JxW_values);
    //
    //         cst n_q_points = quadrature_formula.size();
    //
    //         auto cell = problem.domain.dof_handler.begin_active();
    //         auto endc = problem.domain.dof_handler.end();
    //         for (; cell != endc; ++cell)
    //         {
    //             fe_values .reinit (cell);
    //
    //             for (st q_point = 0; q_point < n_q_points; ++q_point)
    //                 for (st i = 0; i < 4; ++i)
    //                 {
    //                     integ += 
    //                         proble.solution[0](cell->vertex_dof_index(i,0)) *
    //                         fe_values.JxW(q_point);
    //                 };
    //         };
    //     };
    // };
//    printf("%f %f %f\n", problem.meta_coefficient[0],
//                         problem.meta_coefficient[1],
//                         problem.meta_coefficient[2]);


//    {
//       dealii::Vector<double> 
//           grad(problem.system_equations.x.size());
//       {
//           typename dealii::DoFHandler<2>::active_cell_iterator cell =
//               problem.domain.dof_handler.begin_active();
//
//           typename dealii::DoFHandler<2>::active_cell_iterator endc =
//               problem.domain.dof_handler.end();
//
//           std::vector<uint8_t> 
//               divider(problem.system_equations.x.size());
//
//           for (; cell != endc; ++cell)
//           {
//               double tau = 0.0
//               FOR_I(0, 4)
//               {
////                   for (size_t q_point = 0; q_point < 4; ++q_point)
////                   {
////                       FOR_J(0, 2)
////                       {
////                           tau += -fe_values.shape_grad (index_i, q_point)[i] *
////                               this->coefficient[i][material_id] *
////                               fe_values.JxW(q_point);
////                       };
////                   };
//                   grad(cell->vertex_dof_index(i,0)) +=
//                       get_grad<dim>(cell, problem.solution[0], i)[0];
//
//                   divider[cell->vertex_dof_index(i,0)] += 1;
//               // printf("I %d %f\n", cell->vertex_dof_index(i,0),
//               //         get_grad<dim>(cell, problem.solution[0], i)[0]);
//               };
//               FOR_I(0, 4)
//                   if (cell->vertex_dof_index(i,0) == 35)
//                   {
//                       printf("%d %d %d %d %f\n", 
//                               cell->vertex_dof_index(0,0),
//                               cell->vertex_dof_index(1,0),
//                               cell->vertex_dof_index(2,0),
//                               cell->vertex_dof_index(3,0),
//                               get_grad<dim>(cell, problem.solution[0], i)[0]);
//                       break;
//                   };
//           };
//           FOR_I(0, divider.size())
//           {
//               grad(i) /= divider[i];
//               // printf("A %d %f\n", i,
//               //         grad(i));
//               // grad(i)[1] /= divider[i];
//           };
//       };
//       {
//           dealii::DataOut<dim> data_out;
//           data_out.attach_dof_handler (problem.domain.dof_handler);
//
//          char suffix[3] = {'x', 'y', 'z'};
//
//          for (uint8_t i = 0; i < 1; ++i)
//          {
//             data_out.add_data_vector (grad, "grad");
//             data_out.build_patches ();
//
//              std::string file_name = "grad_x";
//              file_name += suffix[i];//i;//boost::lexical_cast<char> (i);
//              file_name += ".gpd";
//
//              std::ofstream output (file_name.data());
//              data_out.write_gnuplot (output);
//          };
//       };
//    };


    std::array<double, 3> meta;
    // meta[1] = max_s;
    meta[0] = problem.meta_coefficient[0];
    meta[1] = problem.meta_coefficient[1];
    meta[2] = problem.meta_coefficient[2];

    printf("meta %lf %lf %lf\n", meta[0], meta[1], meta[2]);

    complete_solution (problem);

    return meta;

};

void complete_solution (HeatConductionProblemOnCell<2> &problem)
{
    enum {x, y, z}; 

    dbl l_xx = problem.meta_coefficient[0];
    dbl l_yy = problem.meta_coefficient[1];
    dbl l_xy = problem.meta_coefficient[2];

    dbl T0_dx = 1.0/(l_xx-(l_xy*l_xy)/l_yy);
    dbl T0_dy = (-l_xy/l_xx)/(l_xx-(l_xy*l_xy)/l_yy);

    dealii::Vector<double> temperature(problem.solution[0].size());
    dealii::Vector<double> heat_conduction[2];
    heat_conduction[x].reinit(problem.solution[0].size());
    heat_conduction[y].reinit(problem.solution[0].size());

    for (st i = 0; i < problem.solution[0].size(); ++i)
    {
        temperature(i) = problem.solution[x](i) * T0_dx + problem.solution[y](i) * T0_dy;
    };

    {
        dealii::DataOut<2> data_out;
        data_out.attach_dof_handler (problem.domain.dof_handler);

        data_out.add_data_vector (temperature, "complete_temperature");
        data_out.build_patches ();

        std::string file_name = "complete_temperature.gpd";

        std::ofstream output (file_name.data());
        data_out.write_gnuplot (output);
    };

    vec<dbl> coef_xx(problem.coefficient[0].size());
    vec<dbl> coef_yy(problem.coefficient[0].size());
    vec<dbl> coef_xy(problem.coefficient[0].size());
    vec<dbl> coef_yx(problem.coefficient[0].size());
    for (st i = 0; i < coef_xx.size(); ++i)
    {
        coef_xx[i] = problem.coefficient[0][i];
        coef_yy[i] = problem.coefficient[0][i];
        coef_xy[i] = problem.coefficient[0][i];
        coef_yx[i] = problem.coefficient[0][i];
    };

    {
        typename dealii::DoFHandler<2>::active_cell_iterator cell =
            problem.domain.dof_handler.begin_active();

        typename dealii::DoFHandler<2>::active_cell_iterator endc =
            problem.domain.dof_handler.end();

        vec<st> divider(problem.solution[0].size());

        for (; cell != endc; ++cell)
        {
            size_t mat_id = cell->material_id();

            arr<prmt::Point<2>, 4> points = {
                cell->vertex(0),
                cell->vertex(1),
                cell->vertex(3),
                cell->vertex(2)};
            arr<dbl, 4> values_x = {
                problem.solution[x](cell->vertex_dof_index (0, 0)),
                problem.solution[x](cell->vertex_dof_index (1, 0)),
                problem.solution[x](cell->vertex_dof_index (3, 0)),
                problem.solution[x](cell->vertex_dof_index (2, 0))};
            arr<dbl, 4> values_y = {
                problem.solution[y](cell->vertex_dof_index (0, 0)),
                problem.solution[y](cell->vertex_dof_index (1, 0)),
                problem.solution[y](cell->vertex_dof_index (3, 0)),
                problem.solution[y](cell->vertex_dof_index (2, 0))};

            Scalar4PointsFunc<2> psi_x(points, values_x);
            Scalar4PointsFunc<2> psi_y(points, values_y);

            for (st i = 0; i < 4; ++i)
            {
                heat_conduction[x](cell->vertex_dof_index(i, 0)) += 
                    (coef_xx[mat_id] * psi_x.dx(points[i]) +
                    coef_xy[mat_id] * psi_x.dy(points[i]) +
                    coef_xx[mat_id]) * T0_dx +
                    (coef_xx[mat_id] * psi_y.dx(points[i]) +
                    coef_xy[mat_id] * psi_y.dy(points[i]) +
                    coef_xy[mat_id]) * T0_dy;

                heat_conduction[y](cell->vertex_dof_index(i, 0)) += 
                    (coef_yx[mat_id] * psi_x.dx(points[i]) +
                    coef_yy[mat_id] * psi_x.dy(points[i]) +
                    coef_yx[mat_id]) * T0_dx +
                    (coef_yx[mat_id] * psi_y.dx(points[i]) +
                    coef_yy[mat_id] * psi_y.dy(points[i]) +
                    coef_yy[mat_id]) * T0_dy;

                ++divider[cell->vertex_dof_index(i, 0)];
            };
        };
        for (st i = 0; i < divider.size(); ++i)
        {
            heat_conduction[x](i) /= divider[i];
            heat_conduction[y](i) /= divider[i];
        };
    };

    {
        dealii::DataOut<2> data_out;
        data_out.attach_dof_handler (problem.domain.dof_handler);

        char suffix[3] = {'x', 'y', 'z'};

        for (uint8_t i = 0; i < 2; ++i)
        {
            data_out.add_data_vector (heat_conduction[x], "h_c");
            data_out.build_patches ();

            std::string file_name = "complete_heat_conduction_";
            file_name += suffix[i];
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };
    };
};

int main(int argc, char *argv[])
{
    const size_t material_id_for_s[2][2] =
    {
        {0, 1},
        {0, 1}
    };

    const size_t material_id_for_line[3][3] =
    {
        {0, 0, 0},
        {1, 1, 1},
        {0, 0, 0}
    };

    const size_t material_id_for_quadrate[4][4] =
    {
        {0, 0, 0, 0},
        {0, 1, 1, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 0}
    };

    const size_t material_id_for_cross[5][5] =
    {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    const size_t material_id_for_shell[6][6] =
    {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0}
    };

    time_t time_1 = time(NULL);
    FOR_J(4, 5)
    {

    // cdbl coef_2 = 393.1;
    // // cdbl coef_1 = 146.538;
    // cdbl coef_1 = 0.030238;

    // dbl coef_1 = 150.0;
    // dbl coef_2 = 1.0;

    dbl coef_1 = 1.0;
    dbl coef_2 = 2.0;

    switch (j)
    {
        case 1:
            {
                FILE *F;
                F = fopen ("mata-quadrate.gpd", "w");

                {
                    // double i = 56.0;//108
                    // while (i < 57.0)
                    FOR (i, 0, 1)
                    {
                        dealii::Triangulation<2> tria;
//
                       const double dot[5] = 
                       {
                           (0.0),
                           (0.25),
                           (0.5),
                           (0.75),
                           (1.0)
                       };
                        // const double dot[3] = 
                        // {
                        //     (0.0),
                        //     (6.795),
                        //     (6.928)
                        // };


                       // ::set_tria <3> (tria, dot, material_id_for_s);
                       ::set_tria <5> (tria, dot, material_id_for_quadrate);
                       // ::set_tria <4> (tria, dot, material_id_for_line);
//                        ::set_line <10> (tria, 1.0, 1.0, 0.5);
//                        ::set_hexagon_grid_pure (tria, 100.0, i);
////                        set_band<2> (tria, 64.0 - 128.0 / 6.0, 64.0 + 128.0 / 6.0, 0);
////                        set_band<2> (tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
//                        //                ::set_quadrate<2>(tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
                       tria .refine_global (2);

                        // ::set_grid(tria);
                        {
                        std::ofstream out ("grid-igor.eps");
                        dealii::GridOut grid_out;
                        grid_out.write_eps (tria , out);
                        };

                        auto res = ::solved<2>(tria, coef_1, coef_2);

                        // fprintf(F, "%f %f %f %f %f %f\n", 
                        //         i*i/16384,
                        //         res[0],
                        //         res[1],
                        //         res[2],
                        //         get_hashin(i*i/16384, coef_1, coef_2),
                        //         get_vanin(i*i/16384, coef_1, coef_2, PI / 2.0, 4)
                        //        );

                        // i += 4;
                        // {
                        //     FILE *F1;
                        //     F1 = fopen ("height4.gpd", "a");
                        //     // fprintf(F1, "%f %f %f\n", coef_2 / coef_1, res[0], res[1]);
                        //     fprintf(F1, "%f %f %f\n", coef_1, res[0], res[1]);
                        //     fclose(F1);
                        // };
                        // coef_2 += 1.0;
        printf("%f %f %f\n", res[0], res[1], res[2]);
                        coef_1 += 1.0;
                        // printf("%f\n", res[0]);
                    };

                };

                fclose(F);
            };
            break;

        case 2:
            {
                FILE *F;
                F = fopen ("mata-cross.gpd", "w");

                {
                    double i = 40.0;//108
                    while (i < 41.0)
                    {
                        dealii::Triangulation<2> tria;

                        const double length = 96.0; //128.0 / 5.0 * 3.0;
                        const double width  = length - sqrt(length*length - i*i); // 128.0 / 5.0;

                        const double dot[6] = 
                        {
                            (0.0),
                            (64.0 - length / 2.0),
                            (64.0 - width  / 2.0),                    
                            (64.0 + width  / 2.0),                
                            (64.0 + length / 2.0),
                            (128.0)
                        };

                        ::set_tria <6> (tria, dot, material_id_for_cross);
                        // tria .refine_global (1);

                        auto res = ::solved<2>(tria, coef_1, coef_2);

                        fprintf(F, "%f %f %f %f %f %f\n", 
                                i*i/16384,
                                res[0],
                                res[1],
                                res[2],
                                get_hashin(i*i/16384, coef_1, coef_2),
                                get_vanin(i*i/16384, coef_1, coef_2, PI / 2.0, 4)
                               );

                        i += 4;
                    };

                };

                fclose(F);
            };
            break;

        case 3:
            {
                FILE *F;
                F = fopen ("mata-shell.gpd", "w");

                {
                    double i = 4.0;//108
                    while (i < 5.0)
                    {
                        dealii::Triangulation<2> tria;

                        const double length = 96.0; //128.0 / 5.0 * 3.0;
                        const double width  = (length - sqrt(length*length - i*i)) / 2.0; // 128.0 / 5.0;

                        const double dot[7] = 
                        {
                            (0.0),
                            (64.0 - length / 2.0),
                            (64.0 - length / 2.0 + width),
                            (64.0),
                            (64.0 + length / 2.0 - width),
                            (64.0 + length / 2.0),
                            (128.0)
                        };

                        ::set_tria <7> (tria, dot, material_id_for_shell);
                        tria .refine_global (4);

                        auto res = ::solved<2>(tria, coef_1, coef_2);

                        fprintf(F, "%f %f %f %f %f %f\n", 
                                i*i/16384,
                                res[0],
                                res[1],
                                res[2],
                                get_hashin(i*i/16384, coef_1, coef_2),
                                get_vanin(i*i/16384, coef_1, coef_2, PI / 2.0, 4)
                               );

                        i += 4;
                    };

                };

                fclose(F);
            };
            break;

        case 4:
            {
                FILE *F;
                F = fopen ("mata-circ.gpd", "w");

                {
                    // double i = 30.0;//108
                    // while (i < 31.0) //68
                    // {
                    double radius = 50.0;
                    double coefic[2] = {1.0, 2.0};
                        dealii::Triangulation<2> tria;

                        ::set_circ(tria, radius, 5); // /sqrt(3.1415926535), 6);

                        auto res = ::solved<2>(tria, coefic[0], coefic[1]);

                        // const double S_I = i*i*PI/16384;

                        // fprintf(F, "%f %f %f %f %f %f\n", 
                        //         S_I,
                        //         res[0],
                        //         res[1],
                        //         res[2],
                        //         get_hashin(S_I, coef_1, coef_2),
                        //         get_vanin(S_I, coef_1, coef_2, PI / 2.0, 4)
                        //        );

                    //     i += 4;
                    // };

                };

                fclose(F);
            };
            break;

        case 5:
            {
                FILE *F;
                F = fopen ("mata-hexagon.gpd", "w");

                {
                    cdbl S   = 83.138438;
                    dbl share = 0.01;
                    // dbl S_I = 81.538438;
                    double i = 50.0;//108
                    while (share < 1.0)
                    {
                        dealii::Triangulation<2> tria;

//                        ::set_hexagon <60> (tria, 100.0, i * (sqrt(3.0) / 2.0));
                        // ::set_hexagon_grid <80> (tria, 100.0, i * (sqrt(3.0) / 2.0));// / sqrt(2.0));
                       // ::set_hexagon_grid_pure (tria, 100.0, i);
                       // ::set_hexagon_grid_pure (tria, 83.138438, 81.538438/83.138438);
                       ::set_hexagon_grid_pure (tria, S, share);
                       // ::set_hexagon_brave <10> (tria, 100.0, 30.0);// / sqrt(2.0));
                       tria .refine_global (4);

                        auto res = ::solved<2>(tria, coef_1, coef_2);

//                         double S = 3.0 * 100.0 * 100.0 * sqrt(3.0);
//                         const double S_I =
//                                   (2 * PI * pow(i * (sqrt(3.0) / 2.0), 2.0)) / S;
// //                                4.0 * 3.14159265359 * 
// //                                pow((i * (sqrt(3.0) / (2.0))), 2.0) / S;

                        fprintf(F, "%f %f %f %f %f %f\n", 
                                (1.0 - share),
                                res[0],
                                res[1],
                                res[2],
                                (S * (1.0 - share)) / 32.0,
                                (coef_1 * (1 - share) + coef_2 * share) 
                                );

                        // fprintf(F, "%f %f %f %f %f %f %f\n", 
                        //         S_I,
                        //         res[0],
                        //         res[1],
                        //         res[2],
                        //         get_hashin(S_I, coef_1, coef_2),
                        //         get_vanin(S_I, coef_1, coef_2, PI / 3.0, 6),
                        //         (coef_1 * (S - S_I) + coef_2 * S_I) / S
                        //        );

                        i += 4;
                        share += 0.01;
                    };

                };

                fclose(F);
            };
            break;

        case 6:
            {
                FILE *F;
                F = fopen ("mata-exper.gpd", "a");

                const size_t material[10][10] =
                {
                    {0,0,0,0,0,0,0,0,0,0},
                    {0,0,0,0,0,0,0,0,0,0},
                    {0,0,0,0,0,0,0,0,0,0},
                    {0,0,0,0,0,0,0,1,0,0},
                    {0,0,0,0,0,0,0,1,0,0},
                    {0,0,0,0,0,0,0,1,0,0},
                    {0,0,0,0,0,0,0,1,0,0},
                    {0,0,0,0,0,0,0,0,0,0},
                    {0,1,1,1,1,0,0,0,0,0},
                    {0,0,0,0,0,0,0,0,0,0}
                };
                dealii::Triangulation<2> tria;

                const double dot[11] = 
                {
                    (0.0),
                    (1.0),
                    (2.0),
                    (3.0),
                    (4.0),
                    (5.0),
                    (6.0),
                    (7.0),
                    (8.0),
                    (9.0),
                    (10.0)
                };

                ::set_tria <11> (tria, dot, material);
                tria .refine_global (2);

                auto res = ::solved<2>(tria, coef_1, coef_2);

                fprintf(F, "%f %f %f %f\n", 
                        res[0],
                        res[1],
                        res[2],
                        (coef_1 * 92.0 + coef_2 * 8.0) / 100.0
                       );

                fclose(F);
            };
            break;
    };
    };

//    {
//        FILE *F;
//        F = fopen ("mata-0.2.gpd", "w");
//        double i = 56.0;
//
//        FOR_K(1, 100)
//        {
//            const double coef_1 = k * 1.0;
//            const double coef_2 = 1.0;
//
//            double a[4] = {0.0};
//
//            FOR_J(1, 5)
//            {
//
//                switch (j)
//                {
//                    case 1:
//                        {
//
//                            dealii::Triangulation<2> tria;
//
//                            const double dot[5] = 
//                            {
//                                (0.0),
//                                (64.0 - i / 2.0),
//                                (64.0),
//                                (64.0 + i / 2.0),
//                                (128.0)
//                            };
//
//                            ::set_tria <5> (tria, dot, material_id_for_quadrate);
//                            tria .refine_global (3);
//
//                            auto res = ::solved<2>(tria, coef_1, coef_2);
//
//                            a[0] = res[0];
//                        };
//                        break;
//
//                    case 2:
//                        {
//                            dealii::Triangulation<2> tria;
//
//                            const double length = 96.0; //128.0 / 5.0 * 3.0;
//                            const double width  = length - sqrt(length*length - i*i); // 128.0 / 5.0;
//
//                            const double dot[6] = 
//                            {
//                                (0.0),
//                                (64.0 - length / 2.0),
//                                (64.0 - width  / 2.0),                    
//                                (64.0 + width  / 2.0),                
//                                (64.0 + length / 2.0),
//                                (128.0)
//                            };
//
//                            ::set_tria <6> (tria, dot, material_id_for_cross);
//                            tria .refine_global (3);
//
//                            auto res = ::solved<2>(tria, coef_1, coef_2);
//
//                            a[1] = res[0];
//                        };
//                        break;
//
//                    case 3:
//                        {
//                            dealii::Triangulation<2> tria;
//
//                            const double length = 96.0; //128.0 / 5.0 * 3.0;
//                            const double width  = (length - sqrt(length*length - i*i)) / 2.0; // 128.0 / 5.0;
//
//                            const double dot[7] = 
//                            {
//                                (0.0),
//                                (64.0 - length / 2.0),
//                                (64.0 - length / 2.0 + width),
//                                (64.0),
//                                (64.0 + length / 2.0 - width),
//                                (64.0 + length / 2.0),
//                                (128.0)
//                            };
//
//                            ::set_tria <7> (tria, dot, material_id_for_shell);
//                            tria .refine_global (3);
//
//                            auto res = ::solved<2>(tria, coef_1, coef_2);
//
//                            a[2] = res[0];
//                        };
//                        break;
//
//                    case 4:
//                        {
//                            dealii::Triangulation<2> tria;
//
//                            ::set_circ(tria, i/sqrt(PI), 6); // /sqrt(3.1415926535), 6);
//
//                            auto res = ::solved<2>(tria, coef_1, coef_2);
//
//                            a[3] = res[0];
//
//                        };
//                };
//            };
//            fprintf(F,"%d %f %f %f %f %f\n",k, a[0], a[1], a[2], a[3],
//                                get_hashin(i*i/16384, coef_1, coef_2));
//        };
//        fclose(F);
//    };


    // printf("%ld\n", (time(NULL) - time_1));
    // 

    // {
    //     std::array<double, 3> l1 = {1./3., 1./3., 1./3.}; 
    //     std::array<double, 3> l2 = {1., 10., 1.}; 
    //     Analit<3> analit(l1, l2);
    //     FILE *F;
    //     F = fopen ("analit.gpd", "w");
    //     double x = 0.0;
    //     while (x < 1.001)
    //     {
    //         fprintf(F, "%f %f\n", x, analit(x));
    //         x += 0.1;
    //     };
    //     fclose(F);
    // };

//    std::vector<std::vector<double> > metas;
//    std::ofstream ofs ("T-qadrate-3.gpd");
////    std::ofstream ofs ("T-shell-3.gpd");
////    metas .push_back (foo(24.0, 104.0, 2));
//    {
//        double i = 40.0;
//        while (i < 44.0)
//        {
//            metas .push_back (foo(65.0 - i / 2.0, 65.0 + i / 2.0, 6));
////            metas .push_back (foo(0.0, i, 3));
//            ofs << i*i << " " << metas.back()[0] << std::endl;
//            i += 4.0;
//        };
//    };
//    for (auto i : metas)
//        printf("%f %f %f\n", i[0], i[1], i[2]);
//    std::cout << metas << std::endl;
    return 0;
}
/////////////////:////////////:////////
