#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vectors.h>
#include <deal.II/numerics/matrices.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

using namespace dealii;

const double L = 4;
const double h = 2;


template <int dim>
class Step4
{
  public:
    Step4 ();
    void run ();

  private:
    void make_grid ();
    void setup_system();
    void assemble_system ();
    void solve ();
    void output_results () const;

    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
};



template <int dim>
class RightHandSide : public Function<dim>
{
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};

template <int dim>
class NeumannBoundaryValues : public Function<dim>
{
  public:
    NeumannBoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
//  double return_value = 0;
//  for (unsigned int i=0; i<dim; ++i)
//    return_value += 4*std::pow(p(i), 4);

//  return -4.0;//-6.0*p(0);//return_value;
//    if ((!((uint16_t)p(0) % 2)))// and (!((uint16_t)p(1) % 2)))
//        return 1.0;
//    else
//        return 2.0;
//  if (sqrt(p(0)*p(0) + p(1)*(1)) < 1.0)
//  if (Point<dim>(2.0,2.0).distance(p) < 1.0)// p.square() < 1.0)
//      return -8.0;
//  else
//      return -8.0;
//  return -8.0;
    return -2.0;
}


template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
//    double ksi = ((p(0) / 2) - floor(p(0) / 2)) * 2;
//
//    if (ksi <= 1)
//        return p(0) + 0.33333333 * ksi;
//    else
//        return p(0) + 0.33333333 * (2 - ksi);
//    if ((abs(p(0) - 0.0) < 1e-5) or (abs(p(0) - L) < 1e-5))
//        return p(0);
//    else 
//        return 0.0;
    return ((p(0)-2)*(p(0)-2) + 1);// ((p(0)-2)*(p(0)-2) + (p(1)-2)*(p(1)-2) + 1);
}

template <int dim>
double NeumannBoundaryValues<dim>::value (const Point<dim> &p,
        const unsigned int /*component*/) const
{
    if ((abs(p(1) - 0.0) < 1e-5) or (abs(p(1) - L) < 1e-5))
        return 0.0;
    else 
        return 0.0;
}

template <int dim>
double coef (const Point<dim> &p, const uint8_t index)
{
//    if ((ceil(p(0)) / 2.0) > 1e5)
//    printf("%d\n", (uint8_t)p(0));
//    if (p(0) < 10.0)
//    if (!((uint16_t)p(0) % 2))
//        return 1.0;
//    else
//        return 2.0;
//    return 1.3333333333;
//    if ((!((uint16_t)p(0) % 2)))// and (!((uint16_t)p(1) % 2)))
//        return 1.0;
//    else
//        return 2.0;
//    return 1.0;
//  return p(0);//p.square();
//    double ksi_x = ((p(0) / h) - floor(p(0) / h)) * h;
//    double ksi_y = ((p(1) / h) - floor(p(1) / h)) * h;
//
//    if (index == 0)
//    {
//        if ((ksi_x > h / 3.0) and (ksi_x < 2*h / 3.0) and
//                (ksi_y > h / 3.0) and (ksi_y < 2*h / 3.0))
//            return 1.0;
//        else
//            return 2.0;
//    }
//    else if (index == 1)
//    {
//        if ((ksi_x > h / 3.0) and (ksi_x < 2*h / 3.0) and
//                (ksi_y > h / 3.0) and (ksi_y < 2*h / 3.0))
//            return 1.0;
//        else
//            return 2.0;
//    };
//  if (sqrt(p(0)*p(0) + p(1)*(1)) < 1.0)
//  if (Point<dim>(2.0,2.0).distance(p) < 1.0)// p.square() < 1.0)
//      return 1.0;
//  else
//      return 2.0;
    if (index == 0)
        return 1.3333333;
    else
        return 1.5000000;
//  if ((p(0) > 1.0) && (p(0) < 3.0))
//      return 1.0;
//  else
//      return 2.0;
//  return 1.746683;
//  return 1.0;
//  return 1.752222;

//    return 1.0;

//    return 0.0;
}





template <int dim>
Step4<dim>::Step4 ()
                :
                fe (1),
                dof_handler (triangulation)
{}



template <int dim>
void Step4<dim>::make_grid ()
{
//  GridGenerator::hyper_cube (triangulation, 1, 2);//10+L);
    std::vector< dealii::Point< 2 > > v (4);

    auto x1(0.0), y1(0.0); 
    auto x2(4.0), y2(4.0); 

    v[0][0] = x1; v[0][1] = y1;
    v[1][0] = x1; v[1][1] = y2;
    v[2][0] = x2; v[2][1] = y1;
    v[3][0] = x2; v[3][1] = y2;

    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());

    c[0].vertices[0] = 2;
    c[0].vertices[1] = 0;
    c[0].vertices[2] = 3;
    c[0].vertices[3] = 1;
    c[0].material_id = 0;

    triangulation .create_triangulation (v, c, dealii::SubCellData());

    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin ();
    cell->face(0) ->set_boundary_indicator (1);
    cell->face(1) ->set_boundary_indicator (1);
    triangulation.refine_global (5);

//    {
//        const Point<2> center (0.0,0.0);
//        const double inner_radius = 0.2,
//              outer_radius = 2.0;
//        GridGenerator::hyper_shell (triangulation,
//                center, inner_radius, outer_radius,
//                10);
//        const HyperShellBoundary<2> boundary_description(center);
//        triangulation.set_boundary (0, boundary_description);
//
//        for (unsigned int step=0; step<5; ++step)
//        {
//            Triangulation<2>::active_cell_iterator
//                cell = triangulation.begin_active(),
//                     endc = triangulation.end();
//
//            for (; cell!=endc; ++cell)
//                for (unsigned int v=0;
//                        v < GeometryInfo<2>::vertices_per_cell;
//                        ++v)
//                {
//                    const double distance_from_center
//                        = center.distance (cell->vertex(v));
//
//                    if (std::fabs(distance_from_center - inner_radius) < 1e-10)
//                    {
//                        cell->set_refine_flag ();
//                        break;
//                    }
//                }
//
//            triangulation.execute_coarsening_and_refinement ();
//        }
//        triangulation.set_boundary (0);
//    };
//    triangulation.refine_global (1);

    std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
}


template <int dim>
void Step4<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  Point<dim> p (0.5);//(20.0, 2.0);
  printf("ТТТ %f\n", dof_handler .get_fe () .shape_value (0, p));

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
  sparsity_pattern.copy_from(c_sparsity);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}



template <int dim>
void Step4<dim>::assemble_system ()
{
    QGauss<dim>  quadrature_formula(2);

    const RightHandSide<dim> right_hand_side;

    FEValues<dim> fe_values (fe, quadrature_formula,
            update_values   | update_gradients |
            update_quadrature_points | update_JxW_values |
            update_jacobians);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
             endc = dof_handler.end();

    bool even = false; 

    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        Point<dim> p (0.5);//(20.0, 2.0);
        //      printf("BBB %f\n", fe_values .get_fe () .shape_value (0, p) * fe_values .jacobian(0)[0][0]);
        printf("BBB %f\n", fe_values .shape_value (0, 1));
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += 
                        (fe_values.shape_grad (i, q_point)[0] *
                         fe_values.shape_grad (j, q_point)[0] *
                         coef<dim>(fe_values.quadrature_point (q_point), 0)
                         +
                         fe_values.shape_grad (i, q_point)[1] *
                         fe_values.shape_grad (j, q_point)[1] *
                         coef<dim>(fe_values.quadrature_point (q_point), 1)
                        ) * fe_values.JxW (q_point);

                cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                        right_hand_side.value (fe_values.quadrature_point (q_point)) *
                        fe_values.JxW (q_point));
            }
        even = not even;

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }


    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
            1,
            BoundaryValues<dim>(),
            boundary_values);
    //  VectorTools::interpolate_boundary_values (dof_handler,
    //                                            0,
    //                                            ConstantFunction<dim>(0.0),
    //                                            boundary_values);
    //  VectorTools::interpolate_boundary_values (dof_handler,
    //                                            1,
    //                                            ConstantFunction<dim>(40.0),
    //                                            boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
            system_matrix,
            solution,
            system_rhs);

    Vector<double> tmp (system_rhs.size());
    VectorTools::create_boundary_right_hand_side (
            dof_handler,
            QGauss<dim-1>(2),
            NeumannBoundaryValues<dim>(),//ConstantFunction<dim>(1),
            tmp);
    system_rhs += tmp;
}



template <int dim>
void Step4<dim>::solve ()
{
  SolverControl           solver_control (100000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}


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

template <int dim>
void Step4<dim>::output_results () const
{
    {
        DataOut<dim> data_out;

        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");

        data_out.build_patches ();

        std::ofstream output ("true-res.gpd");
        data_out.write_gnuplot (output);

        //    std::ofstream output ("true-res.vtk");
        //    data_out.write_vtk (output);
    };

    struct ResType
    {
        dealii::Point<dim, double> point;
        double value;
    };

//    std::vector<ResType> grad_x;
//    std::vector<ResType> grad_y;
    Vector<double> grad(solution.size());

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
             endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            ResType res_x;
            ResType res_y;
            {
                auto temp = get_grad<dim> (cell, solution, i);
                res_x.value = temp[0];
                res_y.value = temp[1];
            };
            res_x.point = cell->vertex(i);
            res_y.point = cell->vertex(i);

//        double ksi_x = ((res_x.point(0) / h) - floor(res_x.point(0) / h)) * h;
//        double ksi_y = ((res_x.point(1) / h) - floor(res_x.point(1) / h)) * h;
//        if ((ksi_x > h / 3.0) and (ksi_x < 2*h / 3.0) and
//                (ksi_y > h / 3.0) and (ksi_y < 2*h / 3.0))
//        {
//            res_x.value *= 1.0;
//            res_y.value *= 1.0;
//        }
//        else
//        {
//            res_x.value *= 10.0;
//            res_y.value *= 10.0;
//        };

        grad(cell->vertex_dof_index(i,0)) = 
            (sqrt(res_x.value * res_x.value + 
                  res_y.value * res_y.value));
        };
    };


    {
        DataOut<dim> data_out;

        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (grad, "grad");

        data_out.build_patches ();

        std::ofstream output ("grad.gpd");
        data_out.write_gnuplot (output);
    };

//    FILE* F;
//    F = fopen ("grad.gpd", "w");
//    for (size_t i = 0; i < grad_x.size(); ++i)
//        fprintf (F, "%f %f %f\n", grad_x[i].point[0], grad_x[i].point[1],
//                (sqrt(grad_x[i].value*grad_x[i].value + grad_y[i].value*grad_y[i].value)));
////    for (auto g : grad)
////        fprintf (F, "%f %f %f\n", g.point[0], g.point[1],
////                g.value);
//    fclose (F);


}




template <int dim>
void Step4<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  make_grid();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}



int main ()
{
//    std::vector<std::unique_ptr<double>> a;
//    double b = 10;
//    a .push_back (std::unique_ptr<double>(b));

    std::vector<double*> a;
    double b = 10;
    a .push_back (&b);

  deallog.depth_console (0);
  {
    Step4<2> laplace_problem_2d;
    laplace_problem_2d.run ();
  }

  return 0;
}

