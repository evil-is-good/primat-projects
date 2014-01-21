/* $Id: step-8.cc 24113 2011-08-19 04:39:39Z bangerth $ */
/* Author: Wolfgang Bangerth, University of Heidelberg, 2000 */

/*    $Id: step-8.cc 24113 2011-08-19 04:39:39Z bangerth $       */
/*                                                                */
/*    Copyright (C) 2000, 2001, 2002, 2003, 2004, 2006, 2007, 2008, 2010, 2011 by the deal.II authors */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

                                 // @sect3{Include files}

				 // As usual, the first few include
				 // files are already known, so we
				 // will not comment on them further.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vectors.h>
#include <deal.II/numerics/matrices.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

				 // In this example, we need
				 // vector-valued finite elements. The
				 // support for these can be found in
				 // the following include file:
#include <deal.II/fe/fe_system.h>
				 // We will compose the vector-valued
				 // finite elements from regular Q1
				 // elements which can be found here,
				 // as usual:
#include <deal.II/fe/fe_q.h>

				 // This again is C++:
#include <fstream>
#include <iostream>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
//#include <emmintrin.h>


//#include </home/primat/projects/common/common.h>
#include </home/primat/projects/deal/main/domain_looper/sources/domain_looper.h>
//#include </home/primat/projects/deal/main/function/function.h>
//#include </home/primat/projects/deal/main/element_stiffness_matrix/element_stiffness_matrix.desc.h>
#include </home/primat/projects/deal/tests/esm_elastic_problem/esm_elastic_problem.h>
#include </home/primat/projects/deal/main/solver_se/solver_se.h>
//#include </home/primat/projects/deal/main/problem/problem.h>
#include </usr/local/include/boost/math/special_functions/sign.hpp>
#include </usr/local/include/boost/lexical_cast.hpp>

				 // The last step is as in previous
				 // programs. In particular, just like in
				 // step-7, we pack everything that's specific
				 // to this program into a namespace of its
				 // own.

// триангуляция
// граничные условия
// тип конечных элементов
// коэффициенты
// элемент матрици
// элемент вектора свободных членов
// 

namespace Step8
{
  using namespace dealii;
  template <int dim, int spacedim>
  class ElasticProblem
  {
    public:
      ElasticProblem ();
      ~ElasticProblem ();
      void run ();

    private:
      void setup_system ();
      void assemble_system ();
      void solve ();
      void refine_grid ();
      void output_results (const unsigned int cycle) const;

      Triangulation<dim, spacedim>   triangulation;
      DoFHandler<dim, spacedim>      dof_handler;

      FESystem<dim, spacedim>        fe;

//      ConstraintMatrix     hanging_node_constraints;

      SparsityPattern      sparsity_pattern;
      SparseMatrix<double> system_matrix;

      Vector<double>       solution;
      Vector<double>       system_rhs;
  };

template <int dim>
class BoundaryValues : public dealii::Function<dim>
{
  public:
    BoundaryValues ();

    virtual void vector_value (const dealii::Point<dim> &p,
                               dealii::Vector<double>   &values) const;
};

template <int dim>
BoundaryValues<dim>::BoundaryValues ()
    :
        dealii::Function<dim> (dim)
{}

template <int dim>
void BoundaryValues<dim>::vector_value (const dealii::Point<dim> &p,
                                        dealii::Vector<double>   &values) const
{
    for (size_t i = 0; i < dim; ++i)
        values(i) = p(i);//15.0;//p(i);
};

  template <int dim>
  class RightHandSide :  public Function<dim>
  {
    public:
      RightHandSide ();
				       
      virtual void vector_value (const Point<dim> &p,
				 Vector<double>   &values) const;

      virtual void vector_value_list (const std::vector<Point<dim> > &points,
				      std::vector<Vector<double> >   &value_list) const;
  };
				  
  template <int dim>
  RightHandSide<dim>::RightHandSide ()
		  :
		  Function<dim> (dim)
  {}
				   
  template <int dim>
  inline
  void RightHandSide<dim>::vector_value (const Point<dim> &p,
					 Vector<double>   &values) const
  {
    Assert (values.size() == dim,
	    ExcDimensionMismatch (values.size(), dim));
    Assert (dim >= 2, ExcNotImplemented());

				    
//    Point<dim> point_1, point_2;
//    point_1(0) = 0.5;
//    point_2(0) = -0.5;
//
//				    
//    if (((p-point_1).square() < 0.2*0.2) ||
//	((p-point_2).square() < 0.2*0.2))
//      values(0) = 1;
//    else
//      values(0) = 0;
//
//    if (p.square() < 0.2*0.2)
//      values(1) = 1;
//    else
//      values(1) = 0;
    values(0) = 0.0;
    values(1) = 0.0;
  }


/////////////:/
  template <int dim>
  void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
					      std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
	    ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();
    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value (points[p],
					value_list[p]);
  }



  template <int dim, int spacedim>
  ElasticProblem<dim, spacedim>::ElasticProblem ()
		  :
		  dof_handler (triangulation),
//		  fe (FE_Q<dim, spacedim>(1), 3)
		  fe (FE_Q<dim, spacedim>(1), dim)
  {}

  template <int dim, int spacedim>
  ElasticProblem<dim, spacedim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }

  int foo()
  {
      srand (time (NULL));
      int res = rand() % 10;
      return res;
  };

template<int d>
int foo()
{
    return d;
}

template <int dim>
int bar(typename DoFHandler<dim>::active_cell_iterator &cell)
{
    return cell->vertex_dof_index(0,0);
};

template<class type, int dim>
type fooo(const Point<dim> &p)
{
    return 0.0;
};


  template <int dim, int spacedim>
  void ElasticProblem<dim, spacedim>::setup_system ()
  {
//      class Femenist::Function<double, dim> ab;
//      Point<dim> p(1,2);
//      ab(p);
//      double (*test)(const Point<dim> &p);
//      test = fooo<double, dim>;
/////////////////////////////////////////////////
    dof_handler.distribute_dofs (fe);
//    unsigned __int128 abcd;
//    size_t x = foo();// dof_handler.n_boundary_dofs();
////    std::array<int,foo()> a;
//    typename DoFHandler<dim>::active_cell_iterator m[foo()];
//    {
//        typename DoFHandler<dim>::active_cell_iterator cell = 
//            dof_handler.begin_active ();
////        std::cout << cell->vertex(0) << std::endl;//face(0)->abc();
////        std::cout << cell->vertex_dof_index(0,1) << std::endl;
////        std::cout << cell->at_boundary() << std::endl;
//        m[0] = cell;
//        ++cell;
//        m[1] = cell;
//        std::cout << "bar " << bar <dim> (cell) << std::endl;
//        int dpc = fe.dofs_per_cell;
//        std::vector<unsigned int> v(dpc);
////        cell->get_dof_indices(v);
////        std::cout << v[0] << std::endl;
////        std::cout << v[1] << std::endl;
////        std::cout << v[2] << std::endl;
////        v[1] = 3;
////        std::cout << v[1] << std::endl;
////        cell->set_dof_indices(v);
////        cell->update_cell_dof_indices_cache();
////        cell->get_dof_indices(v);
////        std::cout << v[0] << std::endl;
////        std::cout << v[1] << std::endl;
////        std::cout << v[2] << std::endl;
//    }
//    std::cout << "true " << foo<true>() << std::endl;
//    std::cout << ((m[0]))->vertex(0) << std::endl;
    
//    std::cout << ((m[1]))->vertex(0) << std::endl;
//    hanging_node_constraints.clear ();
//    DoFTools::make_hanging_node_constraints (dof_handler,
//					     hanging_node_constraints);
//    hanging_node_constraints.close ();
      const bool vector_space = true;
        BlackOnWhiteSubstituter black_on_white_substituter;
///////////////////////////////////////////////////////////////////////////////////////////////////
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
//    c_sparsity .add(0,18);
//    c_sparsity .add(18,0);
    {
        DomainLooper<dim, vector_space> dl;
        REPORT dl .loop_domain(dof_handler, black_on_white_substituter, c_sparsity);
    }
    std::ofstream output ("csp.out");
    c_sparsity .print_gnuplot (output);
    
    printf("11111111111111111\n" );
//    c_sparsity.compress ();
    printf("22222111111111111\n" );
    sparsity_pattern.copy_from(c_sparsity);
    printf("33333111111111111\n" );
    
//  sparsity_pattern.reinit (dof_handler.n_dofs(),
//			     dof_handler.n_dofs(),
//			     dof_handler.max_couplings_between_dofs());
//    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
//
//    hanging_node_constraints.condense (sparsity_pattern);
//
//    std::ofstream output ("matrix.out");
////    sparsity_pattern.compress();
//    sparsity_pattern .print_gnuplot (output);

    system_matrix.reinit (sparsity_pattern);
///////////////////////////////////////////////:////:///////////////////////////////////////
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    double a = (sqrt((double)(8 * 10 + 1)) - 1) / 2;
    size_t b = (size_t)a;
    if (abs(trunc(a) - a) < 0.00000000001)
        printf("TRUE\n");
  }

template <int dim, int spacedim>
void ElasticProblem<dim, spacedim>::assemble_system ()
{
    QGauss<dim>  quadrature_formula(2);

    FEValues<dim, spacedim> fe_values (fe, quadrature_formula,
            update_values   | update_gradients |
            update_quadrature_points | update_JxW_values);

    std::ofstream fout ("1.out");
    fout << fe.dofs_per_cell << std::endl;

    //    TestESM<dim> test_esm;
    //    double coef[2] = {1.0, 1.0};
    //    test_esm .set_coefficient(coef);

    ElementStiffnessMatrixElasticProblem<dim> esm_ep;

    typename ElasticProblemSup<dim>::TypeCoef coef;
    Femenist::Function<std::array<double, dim>, dim> rhsv;
    Femenist::Function<std::array<double, dim>, dim> bound;

    double lambd = 1.0;
    double m     = 1.0;

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    coef[i][j][k][l] .resize (1);
    
    coef[0][0][0][0][0] = lambd + 2 * m;
    coef[1][1][1][1][0] = lambd + 2 * m;

    coef[0][0][1][1][0] = lambd;
    coef[1][1][0][0][0] = lambd;

    coef[0][1][0][1][0] = m;
    coef[1][0][1][0][0] = m;
    coef[0][1][1][0][0] = m;
    coef[1][0][0][1][0] = m;

    esm_ep .set_coefficient (coef);

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    printf("coef[%ld][%ld][%ld][%ld] = %f\n",
                            i,j,k,l,coef[i][j][k][l][0]);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    std::vector<double>     lambda_values (n_q_points);
    std::vector<double>     mu_values (n_q_points);

    ConstantFunction<spacedim> lambda(1.), mu(1.);

    RightHandSide<spacedim>      right_hand_side;
    std::vector<Vector<double> > rhs_values (n_q_points,
            Vector<double>(dim));

    typename DoFHandler<dim, spacedim>::active_cell_iterator cell = dof_handler.begin_active(),
             endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
        mu.value_list     (fe_values.get_quadrature_points(), mu_values);

        right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                rhs_values);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int
                component_i = fe.system_to_component_index(i).first;

            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int
                    component_j = fe.system_to_component_index(j).first;

                //              printf("i = %d, j = %d, res = %f\n", i, j,
                //                      test_esm (i, j, quadrature_formula, fe_values));

                //                cell_matrix(i,j) += test_esm (i, j,
                //                        quadrature_formula, fe_values);
                ////
                double res = 0;
                for (unsigned int q_point=0; q_point<n_q_points;
                        ++q_point)
                {
                    cell_matrix(i,j)
//                    res
                        +=
                        (
                         (fe_values.shape_grad(i,q_point)[component_i] *
                          fe_values.shape_grad(j,q_point)[component_j] *
                          lambda_values[q_point])
                         +
                         (fe_values.shape_grad(i,q_point)[component_j] *
                          fe_values.shape_grad(j,q_point)[component_i] *
                          mu_values[q_point])
                         +
                         ((component_i == component_j) ?
                          (fe_values.shape_grad(i,q_point) *
                           fe_values.shape_grad(j,q_point) *
                           mu_values[q_point])  :
                          0)
                        )
                        *
                        fe_values.JxW(q_point);
                    ////              printf("res=%f\n", res);

                }
//                printf("EC %d %d %f %f\n", i, j, cell_matrix(i,j), esm_ep(i, j,
//                        quadrature_formula, fe_values, 0) / 2.0);
            }
        }
        printf("!!!!!!!!!!!!!\n");

//        for (unsigned int i=0; i<dofs_per_cell; ++i)
//        {
//            const unsigned int
//                component_i = fe.system_to_component_index(i).first;
//
//            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
//                cell_rhs(i) += fe_values.shape_value(i,q_point) *
//                    rhs_values[q_point](component_i) *
//                    fe_values.JxW(q_point);
//        }
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
    for (size_t i = 0; i < system_matrix.m(); ++i)
        for (size_t j = 0; j < system_matrix.m(); ++j)
            if (system_matrix.el(i,j))
                printf("A[%ld][%ld]=%f\n", i,j,system_matrix.el(i,j));

    //    hanging_node_constraints.condense (system_matrix);
    //    hanging_node_constraints.condense (system_rhs);
//    for (size_t i = 0; i < system_matrix.m(); ++i)
//        for (size_t j = 0; j < system_matrix.n(); ++j)
//            if (system_matrix .el (i,j))
//                printf("system_matrix_old(%ld,%ld)=%f\n", i,j,system_matrix(i,j));

//    ConstantFunction<dim> ccc(0,dim);

    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
            0,
//            ccc,
//            ConstantFunction<dim>(10.0, dim),// 
            //ZeroFunction<dim>(dim),
            BoundaryValues<2>(),
            boundary_values);
        printf("???????????\n");
    MatrixTools::apply_boundary_values (boundary_values,
            system_matrix,
            solution,
            system_rhs);


    SolverControl           s_c (1000, 1e-4);
    printf("%d %d %d\n", 
            s_c.check(500, 0.5),
            s_c.check(1001,0.5),
            s_c.check(500,0.000001));
    //    using boost::lexical_cast;
    std::string str;
    str = "aaa"; 
    str += "bbb";
    str += boost::lexical_cast<std::string>(0.3);
    std::cout << str << std::endl;
}



	
  template <int dim, int spacedim>
  void ElasticProblem<dim, spacedim>::solve ()
  {
    SolverControl           solver_control (10000, 1e-12);
//    Problem<dim,int> problem;

//        for (size_t i = 0; i < system_matrix.m(); ++i)
//            for (size_t j = 0; j < system_matrix.n(); ++j)
//                if (system_matrix .el (i,j))
//                printf("system_matrix(%d,%d)=%f\n", i,j,system_matrix(i,j));
////////
    if (0)
    {
        SolverCG<> cg (solver_control);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.0);

        cg.solve (system_matrix, solution, system_rhs,
                preconditioner);
    }
    else
    {
        Femenist::SolverSE<> solver(solver_control);
        REPORT solver .solve (system_matrix, solution, system_rhs);
    }

    printf("%d %f\n", solver_control.last_step(),
            solver_control.last_value());

//    hanging_node_constraints.distribute (solution);
  }

  template <int dim, int spacedim>
  void ElasticProblem<dim, spacedim>::refine_grid ()
  {
//    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
//
//    typename FunctionMap<dim>::type neumann_boundary;
//    KellyErrorEstimator<dim>::estimate (dof_handler,
//					QGauss<dim-1>(2),
//					neumann_boundary,
//					solution,
//					estimated_error_per_cell);
//
//    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
//						     estimated_error_per_cell,
//						     0.3, 0.03);
//
//    triangulation.execute_coarsening_and_refinement ();
  }


		  template <int dim, int spacedim>
  void ElasticProblem<dim, spacedim>::output_results (const unsigned int cycle) const
  {
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".gmv";
    std::ofstream output (filename.c_str());

    DataOut<dim, DoFHandler<dim,spacedim> > data_out;
    data_out.attach_dof_handler (dof_handler);



    std::vector<std::string> solution_names;
    switch (dim)
    {
        case 1:
            solution_names.push_back ("displacement");
            break;
        case 2:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            break;
        case 3:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            solution_names.push_back ("z_displacement");
            break;
        default:
            Assert (false, ExcNotImplemented());
    }

    data_out.add_data_vector (solution, solution_names, 
            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
    data_out.build_patches ();
    data_out.write_gnuplot (output);
  }


  template <int dim, int spacedim>
  void ElasticProblem<dim, spacedim>::run ()
  {
    for (unsigned int cycle=0; cycle<1; ++cycle)
      {
	std::cout << "Cycle " << cycle << ':' << std::endl;

	if (cycle == 0)
	  {
	    GridGenerator::hyper_cube (triangulation, 0, 4);
	    triangulation.refine_global (2);
	  }
	else
	  refine_grid ();

	std::cout << "   Number of active cells:       "
		  << triangulation.n_active_cells()
		  << std::endl;

    printf("step-1\n");
	setup_system ();

	std::cout << "   Number of degrees of freedom: "
		  << dof_handler.n_dofs()
		  << std::endl;

    printf("step-2\n");
	assemble_system ();
    printf("step-3\n");
	solve ();
    printf("step-4\n");
	output_results (0);
      }
  }
}

int main ()
{
  try
    {
      dealii::deallog.depth_console (0);

      Step8::ElasticProblem<2, 2> elastic_problem_2d;
      elastic_problem_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }

  return 0;
}
