/*
 * =====================================================================================
 *
 *       Filename:  heat.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  17.09.2012 10:17:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <projects/deal/tests/heat_conduction_problem/heat_conduction_problem.h>
#include <projects/deal/tests/heat_conduction_problem_on_cell/heat_conduction_problem_on_cell.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <algorithm>
#include <functional>

const double L = 4.0;
const double period = 4.0;

template<uint8_t dim>
dealii::Point<dim, double> get_ksi (const dealii::Point<dim, double> &T0_point)
{
    dealii::Point<dim, double> ksi;

    for (uint8_t i = 0; i < dim; ++i)
        ksi(i) = ((T0_point(i) / period) - floor(T0_point(i) / period)) * period;

    return ksi;
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

template<uint8_t dim>
dealii::Point<dim, double> get_value (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> *solution,
        const dealii::Point<dim, double> &p)
{
    dealii::Point<dim, double> value;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    for (uint8_t i = 0; i < dim; ++i)
    {
        double f1 = solution[i](cell->vertex_dof_index (0, 0));
        double f2 = solution[i](cell->vertex_dof_index (1, 0));
        double f3 = solution[i](cell->vertex_dof_index (2, 0));
        double f4 = solution[i](cell->vertex_dof_index (3, 0));

        double a=-(x2*y2*x3*y4*f1+y1*x3*x4*y4*f2-y1*x3*x2*y2*f4-x1*y1*x3*y4*f2-x1*y3*x4*y4*f2-x1*f3*x2*y2*y4+x1*x3*y3*f2*y4+x1*y3*x2*y2*f4+y1*f3*x4*x2*y2-y3*x4*x2*y2*f1-y3*y1*x3*x4*f2+x1*y3*x4*y1*f2-x2*y3*x3*y4*f1+x1*y1*f3*y4*x2+y1*y3*x3*x2*f4+x2*y3*x4*y4*f1-y1*f3*y4*x4*x2+x3*y3*x4*y2*f1-x3*f1*y2*x4*y4+y2*x3*x1*y1*f4+y2*x1*f3*y4*x4-y2*x4*x1*y1*f3-y2*x1*y3*x3*f4-x1*y1*y3*x2*f4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

        value(i) = a + b * p(0) + c * p(1) + d * p(0) * p(1);
    };

    return value;
};

template<uint8_t dim>
bool contains (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const typename dealii::Point<dim, double> &p)
{
    const dealii::Point<dim, double> p1 = cell->vertex(0);
    const dealii::Point<dim, double> p2 = cell->vertex(1);
    const dealii::Point<dim, double> p3 = cell->vertex(2);
    const dealii::Point<dim, double> p4 = cell->vertex(3);

    bool res = false;

//    std::function<bool (
//            const dealii::Point<dim, double>, 
//            const dealii::Point<dim, double>
//            )>
        auto above_the_line = 
        [p] (const dealii::Point<dim, double> pl, 
             const dealii::Point<dim, double> pr) -> bool
        {
            const uint8_t x = 0;
            const uint8_t y = 1;

//    printf("%f %f %f %f %f\n", 
//            pl(x), pl(y), pr(x), pr(y),(
//                    pl(x) * pr(y) - 
//                    pr(x) * pl(y) + 
//                    p(x)  * (pl(y) - pr(y)) + 
//                    p(y)  * (pr(x) - pl(x))));

            if ((
                    pl(x) * pr(y) - 
                    pr(x) * pl(y) + 
                    p(x)  * (pl(y) - pr(y)) + 
                    p(y)  * (pr(x) - pl(x))) >= -1e-12)
                return true;
            else
                return false;
        };

    if (above_the_line (p1, p4))
    {
        if (above_the_line (p4, p3) and above_the_line (p3, p1))
            res = true;
    }
    else
    {
        if (above_the_line (p1, p2) and above_the_line (p2, p4))
            res = true;
    };

//    if (res)
//        if ((fabs(p1(x)) < 1e-1) and (fabs(p1(y)) < 1e-1))
//    {
//    printf("%f %f %f %f %f %f %f %f %f %f\n", p1(x), p1(y), p2(x),
//            p2(y), p3(x), p3(y), p4(x), p4(y), p(x), p(y));
//    printf("%d %d %d\n", 
//            above_the_line(p1,p4),
//            above_the_line(p4,p3),
//            above_the_line(p3,p1));
//    };

    return res;
};


template<int dim>
double boundary (const dealii::Point<dim> &p)
{
    double ksi = ((p(0) / 2) - floor(p(0) / 2)) * 2;

    if (ksi <= 1)
        return p(0) + 0.33333333 * ksi;
    else
        return p(0) + 0.33333333 * (2 - ksi);
//    return p(0);//(p(0)*p(0))*(p(1)*p(1));
};

template<int dim>
double source (const dealii::Point<dim> &p)
{
//    return 0.0;//-2.0*((p(0)*p(0)) + (p(1)*p(1)));
//  if (dealii::Point<dim>(1.0,1.0).distance(p) < 1.0)// p.square() < 1.0)
//      return -8.0;
//  else
//      return -8.0;
//  return -4.0;
    return -4.0;
};

//template <int dim>
//class DirichletBoundaryValues : public dealii::Function<dim>
//{
//  public:
//      DirichletBoundaryValues () : dealii::Function<dim>() {}
//
//    virtual double value (const dealii::Point<dim>   &p,
//                          const unsigned int  component = 0) const;
//};
//
//template <int dim>
//class NeumannBoundaryValues : public dealii::Function<dim>
//{
//  public:
//    NeumannBoundaryValues () : dealii::Function<dim>() {}
//
//    virtual double value (const dealii::Point<dim>   &p,
//                          const unsigned int  component = 0) const;
//};
//
//template <int dim>
//double DirichletBoundaryValues<dim>::value (const dealii::Point<dim> &p,
//                                   const unsigned int /*component*/) const
//{
//    return p(0);
//};
//
//template <int dim>
//double NeumannBoundaryValues<dim>::value (const dealii::Point<dim> &p,
//        const unsigned int /*component*/) const
//{
////    if ((abs(p(1) - 0.0) < 1e-5) or (abs(p(1) - L) < 1e-5))
////        return 1.0;
////    else 
//    return 0.0;
//};

template <uint8_t dim>
struct Point
{
    double coor[dim];
    double value;
};

template <uint8_t dim>
void get_data (std::vector<Point<dim> > &points, 
        char* file_name)
{
    FILE *F;
    F = fopen (file_name, "r");
    {
        char ch[100];
        for (int i = 0; i < 7; ++i)
            fgets (ch, 100, F);
    };

    Point<dim> point;

    fscanf (F, "%lf %lf %lf", &point.coor[0], &point.coor[1], &point.value);
    points .push_back (point);

    while (fscanf (F, "%lf %lf %lf", &point.coor[0], &point.coor[1],  &point.value) !=
            EOF)
    {
        if (fabs(point.coor[0] - points.back().coor[0]) > 1e-10)
            points .push_back (point);
    };


    fclose (F);

};

struct Neighbor
{
    Neighbor () : distance_x(0.0), distance_y(0.0) {};
    double along_x[2];
    double along_y[2];
    double distance_x;
    double distance_y;
};

int main(int argc, char *argv[])
{
    const uint8_t dim = 2;

    const uint8_t xx = 0;
    const uint8_t yy = 1;
    const uint8_t xy = 2;

//    const double L = 4.0;

    ///////////////Solved on cell

    std::array<std::vector<double>, 3> coef_on_cell;

    coef_on_cell[xx] .resize (2);
    coef_on_cell[yy] .resize (2);
    coef_on_cell[xy] .resize (2);

    coef_on_cell[xx][0] = 2.0; coef_on_cell[xx][1] = 1.0;
    coef_on_cell[yy][0] = 2.0; coef_on_cell[yy][1] = 1.0;
    coef_on_cell[xy][0] = 0.0; coef_on_cell[xy][1] = 0.0;

    dealii::Triangulation<dim> triangulation_cell;

//    std::vector< dealii::Point< 2 > > v (6);
//
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 1.0; v[1][1] = 0.0;
//    v[2][0] = 2.0; v[2][1] = 0.0;
//    v[3][0] = 0.0; v[3][1] = 2.0;
//    v[4][0] = 1.0; v[4][1] = 2.0;
//    v[5][0] = 2.0; v[5][1] = 2.0;
//
//    std::vector< dealii::CellData< 2 > > c (2, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 3;
//    c[0].vertices[3] = 4;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 4;
//    c[1].vertices[3] = 5;
//    c[1].material_id = 1;

//    std::vector< dealii::Point< 2 > > v (8);
//
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 4.0; v[1][1] = 0.0;
//    v[2][0] = 1.0; v[2][1] = 1.0;
//    v[3][0] = 3.0; v[3][1] = 1.0;
//    v[4][0] = 0.0; v[4][1] = 4.0;
//    v[5][0] = 1.0; v[5][1] = 3.0;
//    v[6][0] = 3.0; v[6][1] = 3.0;
//    v[7][0] = 4.0; v[7][1] = 4.0;
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

//    std::vector< dealii::Point< 2 > > v (9);
//
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 1.0; v[1][1] = 0.0;
//    v[2][0] = 2.0; v[2][1] = 0.0;
//    v[3][0] = 0.0; v[3][1] = 1.0;
//    v[4][0] = 1.0; v[4][1] = 1.0;
//    v[5][0] = 2.0; v[5][1] = 1.0;
//    v[6][0] = 0.0; v[6][1] = 2.0;
//    v[7][0] = 1.0; v[7][1] = 2.0;
//    v[8][0] = 2.0; v[8][1] = 2.0;
//
//    std::vector< dealii::CellData< 2 > > c (4, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 3;
//    c[0].vertices[3] = 4;
//    c[0].material_id = 1;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 4;
//    c[1].vertices[3] = 5;
//    c[1].material_id = 0;
//    
//    c[2].vertices[0] = 3;
//    c[2].vertices[1] = 4;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 4;
//    c[3].vertices[1] = 5;
//    c[3].vertices[2] = 7;
//    c[3].vertices[3] = 8;
//    c[3].material_id = 0;
    
    /////////////////////////////////////////////////////
//    std::vector< dealii::Point< 2 > > v (16);
//
//    double a1 = 0.0;
//    double a2 = 2.0 / 3.0;
//    double a3 = 4.0 / 3.0;
//    double a4 = 2.0;
//
//    v[0][0] = a1; v[0][1] = a1;
//    v[1][0] = a2; v[1][1] = a1;
//    v[2][0] = a3; v[2][1] = a1;
//    v[3][0] = a4; v[3][1] = a1;
//    v[4][0] = a1; v[4][1] = a2;
//    v[5][0] = a2; v[5][1] = a2;
//    v[6][0] = a3; v[6][1] = a2;
//    v[7][0] = a4; v[7][1] = a2;
//    v[8][0] = a1; v[8][1] = a3;
//    v[9][0] = a2; v[9][1] = a3;
//    v[10][0] = a3; v[10][1] = a3;
//    v[11][0] = a4; v[11][1] = a3;
//    v[12][0] = a1; v[12][1] = a4;
//    v[13][0] = a2; v[13][1] = a4;
//    v[14][0] = a3; v[14][1] = a4;
//    v[15][0] = a4; v[15][1] = a4;
//
//    std::vector< dealii::CellData< 2 > > c (9, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 4;
//    c[0].vertices[3] = 5;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 5;
//    c[1].vertices[3] = 6;
//    c[1].material_id = 0;
//    
//    c[2].vertices[0] = 2;
//    c[2].vertices[1] = 3;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 4;
//    c[3].vertices[1] = 5;
//    c[3].vertices[2] = 8;
//    c[3].vertices[3] = 9;
//    c[3].material_id = 0;
//    
//    c[4].vertices[0] = 5;
//    c[4].vertices[1] = 6;
//    c[4].vertices[2] = 9;
//    c[4].vertices[3] = 10;
//    c[4].material_id = 1;
//    
//    c[5].vertices[0] = 6;
//    c[5].vertices[1] = 7;
//    c[5].vertices[2] = 10;
//    c[5].vertices[3] = 11;
//    c[5].material_id = 0;
//    
//    c[6].vertices[0] = 8;
//    c[6].vertices[1] = 9;
//    c[6].vertices[2] = 12;
//    c[6].vertices[3] = 13;
//    c[6].material_id = 0;
//    
//    c[7].vertices[0] = 9;
//    c[7].vertices[1] = 10;
//    c[7].vertices[2] = 13;
//    c[7].vertices[3] = 14;
//    c[7].material_id = 0;
//    
//    c[8].vertices[0] = 10;
//    c[8].vertices[1] = 11;
//    c[8].vertices[2] = 14;
//    c[8].vertices[3] = 15;
//    c[8].material_id = 0;
//    
//    triangulation_cell .create_triangulation (v, c, dealii::SubCellData());

//    dealii::GridGenerator::hyper_cube (triangulation_cell , 0, L);
//    triangulation_cell .refine_global (6);
//
//    {
//        dealii::Point<dim> center (2.0,2.0);
//        dealii::Triangulation<2>::active_cell_iterator
//            cell = triangulation_cell .begin_active(),
//                 end_cell = triangulation_cell .end();
//        for (; cell != end_cell; ++cell)
//        {
//            dealii::Point<dim> midle_p(0.0, 0.0);
//
//            for (size_t i = 0; i < 4; ++i)
//            {
//                midle_p(0) += cell->vertex(i)(0);
//                midle_p(1) += cell->vertex(i)(1);
//            };
//            midle_p(0) /= 4;
//            midle_p(1) /= 4;
//
////            double ksi_x = ((cell->vertex(i)(0) / period) - floor(cell->vertex(i)(0) / period)) * period;
////            double ksi_y = ((cell->vertex(i)(1) / period) - floor(cell->vertex(i)(1) / period)) * period;
////
////            if ((ksi_x > period / 3.0) and (ksi_x < 2*period / 3.0) and
////                    (ksi_y >  period/ 3.0) and (ksi_y < 2*period / 3.0))
////                cell->set_material_id(1);
////            else
////                cell->set_material_id(0);
////        };
//
//            if (center.distance(midle_p) < 1.0)
//                cell->set_material_id(1);
//            else
//                cell->set_material_id(0);
//        };
//    };

    std::vector< dealii::Point< 2 > > v (8);

    v[0][0] = 0.0; v[0][1] = 0.0;
    v[1][0] = 1.0; v[1][1] = 0.0;
    v[2][0] = 3.0; v[2][1] = 0.0;
    v[3][0] = 4.0; v[3][1] = 0.0;
    v[4][0] = 0.0; v[4][1] = 4.0;
    v[5][0] = 1.0; v[5][1] = 4.0;
    v[6][0] = 3.0; v[6][1] = 4.0;
    v[7][0] = 4.0; v[7][1] = 4.0;

    std::vector< dealii::CellData< dim > > c (3, dealii::CellData<dim>());

    c[0].vertices[0] = 0;
    c[0].vertices[1] = 1;
    c[0].vertices[2] = 4;
    c[0].vertices[3] = 5;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 5;
    c[1].vertices[3] = 6;
    c[1].material_id = 1;

    c[2].vertices[0] = 2;
    c[2].vertices[1] = 3;
    c[2].vertices[2] = 6;
    c[2].vertices[3] = 7;
    c[2].material_id = 0;

    triangulation_cell .create_triangulation (v, c, dealii::SubCellData());

    triangulation_cell .refine_global (4);

//    std::ofstream out ("grid-1.eps");
//    dealii::GridOut grid_out;
//    grid_out.write_eps (triangulation, out);

    class ::HeatConductionProblemOnCell<dim> psi (
            triangulation_cell, coef_on_cell);

    REPORT psi .solved ();

    psi .print_result ("res_on_cell_");


//    printf("%f %f %f\n", problem.mean_coefficient[0],
//                         problem.mean_coefficient[1],
//                         problem.mean_coefficient[2]);
//
//    printf("%f %f %f\n", problem.meta_coefficient[0],
//                         problem.meta_coefficient[1],
//                         problem.meta_coefficient[2]);

     
    ///////////Solved in metamaterial


    typename HeatConductionProblemSup<dim>::TypeCoef coef;
    Femenist::Function<double, dim> rhsv;
//    Femenist::Function<double, dim> bound;

    coef[xx] .push_back (psi.meta_coefficient[xx]);
    coef[yy] .push_back (psi.meta_coefficient[yy]);
    coef[xy] .push_back (psi.meta_coefficient[xy]);
//    coef[xx] .push_back (psi.meta_coefficient[xx]);
//    coef[yy] .push_back (psi.meta_coefficient[yy]);
//    coef[xy] .push_back (psi.meta_coefficient[xy]);

    rhsv    = source<dim>;

//    bound   = boundary<dim>;

    Femenist::MyFuncFromDealii<dim>::Func DirichletBoundaryValues =
        [] (const dealii::Point<dim> &p) //{return p(0);};
//    {return (p(0)-2)*(p(0)-2) + (p(1)-2)*(p(1)-2) + 1;};
    {return (((p(0)>1.0)and(p(0)<3.0))?
        (2*(p(0)-2)*(p(0)-2)):
            ((p(0)-2)*(p(0)-2) + 1));};

    Femenist::MyFuncFromDealii<dim>::Func NeumannBoundaryValues =
        [] (const dealii::Point<dim> &p) {return 0.0;};

    std::vector<Femenist::BoundaryValues<dim> > bound(2);
    bound[0].function = DirichletBoundaryValues;
    bound[0].boundari_indicator = 1;
    bound[0].type = 0;
    bound[1].function = NeumannBoundaryValues;
    bound[1].boundari_indicator = 0;
    bound[1].type = 1;

//    std::vector<Femenist::BoundaryValues<dim> > bound(1);
//    bound[0].function = DirichletBoundaryValues;
//    bound[0].boundari_indicator = 0;
//    bound[0].type = 0;

//    dealii::Point<dim> pp (1.0,0.0);
//    printf("%f\n", bound[0].function.value(pp));
//    printf("%f\n", bound[1].function.value(pp));

//    std::function<double (const dealii::Point<dim>&)> aaa = 
//        [] (const dealii::Point<dim> &p) {return 2.0;};
//
//    MyFunc<dim> mf(aaa);
//    printf("%f\n", mf.value(pp));
    

//    std::vector< dealii::Point< 1 > > v (3);
//
//    v[0][0] = 0.0; 
//    v[1][0] = 1.0;
//    v[2][0] = 2.0;
//
//    std::vector< dealii::CellData< 1 > > c (2, dealii::CellData<1>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].material_id = 1;

    dealii::Triangulation<dim> tria;

    dealii::GridGenerator::hyper_cube (tria, 0, L);//10, 10+L);

//    tria .create_triangulation (v, c, dealii::SubCellData());


  typename dealii::Triangulation<dim>::active_cell_iterator cell 
      = tria.begin ();
    tria.begin()->face(0) ->set_boundary_indicator (1);
    tria.begin()->face(1) ->set_boundary_indicator (1);

    tria .refine_global (4);

    class ::HeatConductionProblem<dim> T0 (tria, coef, bound, rhsv);

    REPORT T0 .solved ();

    T0.system_equations.x(0) = 0.0;//10.0;

    T0 .print_result ("T0.gpd");

    printf("%f %f %f\n", psi.meta_coefficient[0],
                         psi.meta_coefficient[1],
                         psi.meta_coefficient[2]);

//return 0;


    ///////////// Crowning solved

    struct ResType
    {
        dealii::Point<dim, double> point;
        double value;
        bool operator== (const ResType &rs) const
        {
            if((fabs(this->point[0] - rs.point[0]) < 1e-8) and
               (fabs(this->point[1] - rs.point[1]) < 1e-8))
                return true;
            else
                return false;
        };
    };

    std::vector<dealii::Point<dim> > T0_grad_field(T0.system_equations.x.size());
    std::vector<dealii::Point<dim> > T0_grad_2_field(T0.system_equations.x.size());
    std::vector<ResType> T(T0.system_equations.x.size());

    typename dealii::DoFHandler<dim>::active_cell_iterator 
        T0_cell = T0.domain.dof_handler.begin_active();
    typename dealii::DoFHandler<dim>::active_cell_iterator 
        T0_end_cell  = T0.domain.dof_handler.end();

//    {
//        typename dealii::DoFHandler<dim>::active_cell_iterator 
//            _T0_cell = psi.domain.dof_handler.begin_active();
//        typename dealii::DoFHandler<dim>::active_cell_iterator 
//            _end_cell  = psi.domain.dof_handler.end();
//
//        size_t count_a = 0;
//
//        dealii::Point<dim, double> ks (1.375, 0.625); 
//        std::ofstream file ("blabla.out");
//        for (; _T0_cell != _end_cell; ++_T0_cell)
//        {
////            if (
////                    (_T0_cell ->vertex(0)(0) < 1.375) and
////                    (_T0_cell ->vertex(0)(1) < 0.625) and
////                    (_T0_cell ->vertex(3)(0) > 1.375) and
////                    (_T0_cell ->vertex(3)(1) > 0.625)
////               )
////            {
//            if (count_a == 683)
//            printf("%d %f %f 1.375 0.625 %d\n",
//                    count_a, 
//                        _T0_cell ->vertex(0)(0),
//                        _T0_cell ->vertex(0)(1),
//                        ::contains<dim> (_T0_cell, ks));
////            };
//            file << count_a << " " << _T0_cell->vertex(0)(0) << " " <<
//                _T0_cell ->vertex(0)(1) << std::endl;
//            ++count_a;
////            printf("%d", ::contains<dim> (_T0_cell, ks));
//        };
//        printf("\n");
//    };
//
//    return 0;

//    FILE *FF;
//    FF = fopen("test.gmp", "w");
//
//    size_t count_123 = 0;

    {
        std::vector<uint8_t> divider(T0.system_equations.x.size());
        T0_cell = T0.domain.dof_handler.begin_active();
        for (; T0_cell != T0_end_cell; ++T0_cell)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                T0_grad_field[T0_cell->vertex_dof_index(i,0)] +=
                    ::get_grad<dim> (T0_cell, T0.system_equations.x, i); 
                divider[T0_cell->vertex_dof_index(i,0)] += 1;
            };
        };
        for (size_t i = 0; i < divider.size(); ++i)
            T0_grad_field[i] /= divider[i];
    };
//
//    {
//        typename dealii::DoFHandler<dim>::active_face_iterator 
//            T0_face_1 = T0.domain.dof_handler.begin_active_face();
//        typename dealii::DoFHandler<dim>::active_face_iterator 
//            T0_end_face  = T0.domain.dof_handler.end_active_face();
//
//        for (; T0_face_1 != T0_end_face; ++T0_face_1)
//            if (not T0_face_1->at_boundary())
//            {
//                Neighbor nei[2];
//
//                typename dealii::DoFHandler<dim>::active_face_iterator 
//                    T0_face_2 = T0.domain.dof_handler.begin_active_face();
//
//                auto temp_lambda = 
//                    [T0_grad_field, T0_face_1, &T0_face_2] 
//                    (uint8_t index1, uint8_t index2, Neighbor* nei)
//                    {
//                printf("%d\n", T0_face_2->vertex_dof_index(0,0));
//                        if (T0_face_1 ->vertex_dof_index(index1,0) == 
//                                T0_face_2 -> vertex_dof_index(index2,0))
//                        {
//                            auto p_1 = T0_face_1 -> vertex (index1);
//                            auto p_2 = T0_face_2 -> vertex (not index2);
//                            if (fabs(p_2(0) - p_1(0)) < 1e-10)
//                            {
//                                uint8_t n = 1;
//                                if (p_2(n) < (p_1(n) - 1e-10))
//                                    nei[index1].along_y[0] = 
//                                        T0_grad_field[
//                                        T0_face_2->vertex_dof_index(not index2,0)](n);
//                                else
//                                    nei[index1].along_y[1] =
//                                        T0_grad_field[
//                                        T0_face_2->vertex_dof_index(not index2,0)](n);
//
//                                nei[index1].distance_y += fabs(p_2(n) - p_1(n));
//                            }
//                            else if (fabs(p_2(1) - p_1(1)) < 1e-10)
//                            {
//                                uint8_t n = 0;
//                                if (p_2(n) < (p_1(n) - 1e-10))
//                                    nei[index1].along_x[0] =
//                                        T0_grad_field[
//                                        T0_face_2->vertex_dof_index(not index2,0)](n);
//                                else
//                                    nei[index1].along_x[1] =
//                                        T0_grad_field[
//                                        T0_face_2->vertex_dof_index(not index2,0)](n);
//
//                                nei[index1].distance_x += fabs(p_2(n) - p_1(n));
//                            };
//                        };
//                    };
//
//                for (; T0_face_2 != T0_end_face; ++T0_face_2)
//                {
//                    temp_lambda(0,0, and_assigned_to nei);
//                    temp_lambda(0,1, and_assigned_to nei);
//                    temp_lambda(1,0, and_assigned_to nei);
//                    temp_lambda(1,1, and_assigned_to nei);
//                };
//
//        printf("%f %f %f %f\n", 
//                nei[0].distance_x,
//                nei[0].distance_y,
//                nei[1].distance_x,
//                nei[1].distance_y);
//
//                T0_grad_2_field [T0_face_1 -> vertex_dof_index(0,0)] =
//                {
//                    (nei[0].along_x[1] - nei[0].along_x[0]) / nei[0].distance_x,
//                    (nei[0].along_y[1] - nei[0].along_y[0]) / nei[0].distance_y
//                };
//
//                T0_grad_2_field [T0_face_1 -> vertex_dof_index(1,0)] =
//                {
//                    (nei[1].along_x[1] - nei[1].along_x[0]) / nei[1].distance_x,
//                    (nei[1].along_y[1] - nei[1].along_y[0]) / nei[1].distance_y
//                };
//
//                printf("%f %d %f %f\n", 
//                        T0_face_1->vertex(0)(0),
//                        T0_face_1->vertex(0)(1),
//                        T0_grad_2_field [T0_face_1 -> vertex_dof_index(0,0)](0),
//                        T0_grad_2_field [T0_face_1 -> vertex_dof_index(0,0)](1));
//            };
//    };

//    double T

//    psi.solution[0](0) = 5.0;
//    psi.solution[1](0) = 5.0;
        T0_cell = T0.domain.dof_handler.begin_active();
    for (; T0_cell != T0_end_cell; ++T0_cell)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            dealii::Point<dim, double> ksi = 
                ::get_ksi<dim> (T0_cell->vertex (i));

            dealii::Point<dim, double> T0_grad = 
                T0_grad_field[T0_cell->vertex_dof_index(i,0)];
//                ::get_grad<dim> (T0_cell, T0.system_equations.x, i); 

            dealii::Point<dim, double> psi_in_T0 (0.0, 0.0);

            typename dealii::DoFHandler<dim>::active_cell_iterator 
                psi_cell = psi.domain.dof_handler.begin_active();
            typename dealii::DoFHandler<dim>::active_cell_iterator 
                psi_end_cell  = psi.domain.dof_handler.end();

            for (; psi_cell != psi_end_cell; ++psi_cell)
            {
                if (::contains<dim> (psi_cell, ksi))
                {
                    psi_in_T0 = psi_cell->vertex(i);//
                    psi_in_T0 = ::get_value<dim> (psi_cell, psi.solution, ksi);
                    break;
                };
            };

//            T0_grad_field[T0_cell->vertex_dof_index(i,0)] = T0_grad;

            if (fabs(T0_cell->vertex(i)(0) - 1.0) < 1e-5)
                printf("%f %f %f %f %f\n", 
                        T0.system_equations.x (T0_cell->vertex_dof_index (i, 0)),
                        psi_in_T0[0],
                        psi_in_T0[1],
                        T0_grad[0],
                        T0_grad[1]);
            T[T0_cell->vertex_dof_index(i,0)] =
            {
                T0_cell->vertex(i),
                T0.system_equations.x (T0_cell->vertex_dof_index (i, 0)) +
                    (psi_in_T0[0]+0) * T0_grad[0] +
                    (psi_in_T0[1]+0) * T0_grad[1] +
//                    (psi_in_T0[0]+0) * 3
                    psi_in_T0[0] * (psi_in_T0[0]+0) * 3
            };
        };
    };


//    for (auto i : T0_grad_2_field){
//        printf("%f %f\n", i(0), i(1));
//    };

//    return 0;

    std::vector<ResType> grad(T0.system_equations.x.size());
    {
        dealii::Vector<double> T_value(T0.system_equations.x.size());

        {
            size_t c = 0;
            for (auto i : T)
            {
                T_value(c) = i.value;
                c++;
            };
        };

        T0_cell = T0.domain.dof_handler.begin_active();
        for (; T0_cell != T0_end_cell; ++T0_cell)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                auto temp = get_grad<dim> (T0_cell, T_value, i);

                grad[T0_cell->vertex_dof_index(i,0)] =
                {
                    T[T0_cell->vertex_dof_index(i,0)].point,
                    sqrt(temp[0] * temp[0] + temp[1] * temp[1])
                };
            };
        };
    };

//    fclose(FF);

    
//    return 0;

//    std::vector<Point<dim> > brut_force_T;
//    get_data<dim> (brut_force_T, "../../force_solve/sources/true-res.gpd");
//        for (Point<dim> bfT : brut_force_T){
//                printf ("%f %f\n", 
//                        bfT.coor[0],
//                        bfT.coor[1]);
//        };
//
//    std::vector<ResType> difT;
    
//    for (const ResType t : T){
//        for (const Point<dim> bfT : brut_force_T){
////            for (size_t i = 0; i < T.size
////                printf ("%f %f %f %f %f %f\n", 
////                        bfT.coor[0],
////                        bfT.coor[1],
////                        t.point[0],
////                        t.point[1],
////            fabs(bfT.coor[0] - t.point[0]),
////                    fabs(bfT.coor[1] - t.point[1]));
//            if (
//                    (fabs(bfT.coor[0] - t.point[0]) < 1e-8) and
//                    (fabs(bfT.coor[1] - t.point[1]) < 1e-8))
//            {
////                printf ("%f %f %f %f %f %f\n", 
////                        bfT.coor[0],
////                        bfT.coor[1],
////                        t.point[0],
////                        t.point[1],
////            fabs(bfT.coor[0] - t.point[0]),
////                    fabs(bfT.coor[1] - t.point[1]));
//                ResType res;
//                res.point[0] = t.point[0];
//                res.point[1] = t.point[1];
//                res.value = fabs(bfT.value - t.value);// / bfT.value;
//                difT .push_back (res);
//                break;
//
////                printf ("%f %f %f\n", 
////                        res.point[0],
////                        t.point[0],
////                        difT.back().point[0]);
//            };
//        };
//    };
    
//    printf("%d %d %d\n", T.size(), difT.size(), brut_force_T.size());

    {
        dealii::Vector<double> T_value(T0.system_equations.x.size());

        {
            size_t c = 0;
            for (auto i : T)
            {
                T_value(c) = i.value;
                c++;
            };
        };

        dealii::DataOut<dim> data_out;

        data_out.attach_dof_handler (T0.domain.dof_handler);
        data_out.add_data_vector (T_value, "T");

        data_out.build_patches ();

        std::ofstream output ("T.gpd");
        data_out.write_gnuplot (output);
    };

//    {
//        dealii::Vector<double> dif(T0.system_equations.x.size());
//
//        {
//            size_t c = 0;
//            for (auto i : T)
//            {
//                //dif(c) = fabs(i.value - fooo());
//                c++;
//            };
//        };
//
//        dealii::DataOut<dim> data_out;
//
//        data_out.attach_dof_handler (T0.domain.dof_handler);
//        data_out.add_data_vector (dif, "dif");
//
//        data_out.build_patches ();
//
//        std::ofstream output ("dif.gpd");
//        data_out.write_gnuplot (output);
//    };

    {
        dealii::Vector<double> grad_value(T0.system_equations.x.size());

        {
            size_t c = 0;
            for (auto i : grad)
            {
                grad_value(c) = i.value;
                c++;
            };
        };

        dealii::DataOut<dim> data_out;

        data_out.attach_dof_handler (T0.domain.dof_handler);
        data_out.add_data_vector (grad_value, "grad");

        data_out.build_patches ();

        std::ofstream output ("grad.gpd");
        data_out.write_gnuplot (output);
    };

//    FILE *F1;
//    FILE *F2;
//    FILE *F3;
//
//    F1 = fopen ("crow.gpd", "w");
//    F2 = fopen ("T0.gpd", "r");
//    F3 = fopen ("difT.gpd", "w");
//
//    fputs ("# <x> <y> <solution>\n", F1);
//
//    {
//        char ch[100];
//        for (int i = 0; i < 7; ++i)
//            fgets (ch, 100, F2);
//    };
//
//    ResType a;
//
//    uint8_t count = 0;
//
//    while (fscanf (F2, "%lf %lf %lf", 
//                &a.point[0], &a.point[1],  &a.value) !=
//            EOF)
//    {
//        bool there_is = false;
//
////        for (const auto t : T){
//////            if (
//////                    (fabs(a.point[0] - t.point[0]) < 1e-8) and
//////                    (fabs(a.point[1] - t.point[1]) < 1e-8))
////            if (t == a)
////            {
////                fprintf (F1, "%f %f %f\n", 
////                        t.point(0),
////                        t.point(1),
////                        t.value);
////                there_is = true;
////            };
////        };
//
//        auto t = std::find(grad.begin(), grad.end(), a);
//
//        if (t != grad.end())
//        {
//            fprintf (F1, "%f %f %f\n", 
//                    t->point(0),
//                    t->point(1),
//                    t->value);
//        }
//        else
//        {
//            puts("ffff");
////            printf ("%f %f %f\n", 
////                    a.point(0),
////                    a.point(1),
////                    a.value);
//        };
//
//
//
////        for (const auto dift : difT){
////            if (
////                    (fabs(a.point[0] - dift.point[0]) < 1e-8) and
////                    (fabs(a.point[1] - dift.point[1]) < 1e-8))
////            {
////                fprintf (F3, "%f %f %f\n", 
////                        dift.point(0),
////                        dift.point(1),
////                        dift.value);
//////                printf ("%f %f %f\n", 
//////                        dift.point(0),
//////                        dift.point(1),
//////                        dift.value);
////            };
////        };
//
////        if (not there_is)
////        {
////            printf ("%f %f %f\n", 
////                    a.point(0),
////                    a.point(1),
////                    a.value);
//////            fprintf (F1, "%f %f %f\n", 
//////                    a.point(0),
//////                    a.point(1),
//////                    0.0);//a.value);//
////        };
//
//        ++count;
//
//        switch (count)
//        {
//            case 2:
//                fprintf(F1,"\n");
//                fprintf(F3,"\n");
//                break;
//            case 4:
//                fprintf(F1,"\n\n");
//                fprintf(F3,"\n\n");
//                count = 0;
//                break;
//        };
//    };
//
//    fclose (F1);
//    fclose (F2);
//    fclose (F3);




    return 0;
}
//
