/*
 * =====================================================================================
 *
 *       Filename:  heat_conduction_problem.desc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  14.09.2012 10:55:04
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <array>
#include <projects/deal/tests/esm_elastic_problem/esm_elastic_problem.h>
#include <projects/deal/main/problem/problem.h>
#include <projects/deal/main/solver_se/solver_se.h>
#include <projects/deal/main/function/function.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/numerics/vectors.h>
#include <deal.II/numerics/matrices.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <fstream>
#include <iostream>


template <int dim>
class BoundaryValues : public dealii::Function<dim>
{
  public:
    BoundaryValues ();

    virtual void vector_value (const dealii::Point<dim> &p,
                               dealii::Vector<double>   &values) const;

    void set (typename ElasticProblemSup<dim>::TypeFunc &value);

    typename ElasticProblemSup<dim>::TypeFunc content;
};

template<uint8_t dim>
class ElementRightHandSideVectorElasticProblem : 
    public ElementRightHandSideVector<
    dim, double, typename ElasticProblemSup<dim>::TypeFunc>
{
    public:
        ElementRightHandSideVectorElasticProblem ();

    virtual void set_coefficient (typename ElasticProblemSup<dim>::TypeFunc &coef); 

    virtual double operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values) const;
};

template <uint8_t dim>
class ElasticProblem : public Problem< 
                              dim, 
                              ElementStiffnessMatrixElasticProblem<dim>, 
                              ElementRightHandSideVectorElasticProblem<dim> >
{
    public:
        ElasticProblem (const dealii::Triangulation<dim> &triangulation,
                        typename ElasticProblemSup<dim>::TypeCoef &coefficient, 
//                        typename ElasticProblemSup<dim>::TypeFunc &boundary_values,
                        std::vector<typename 
                        ElasticProblemSup<dim>::BoundaryValues> &boundary_values,
                        typename ElasticProblemSup<dim>::TypeFunc &rhs_values);
        ~ElasticProblem ();
    //Methods
    public:
        virtual prmt::Report solved ();
        virtual void print_result (const std::string &filename);

        dealii::FESystem<dim> finite_element;

    protected:
        virtual prmt::Report setup_system ();
        virtual prmt::Report assemble_system ();
        virtual prmt::Report apply_boundary_values ();
        virtual prmt::Report solve_system_equations ();
        virtual prmt::Report output_results ();

    //Fields
    protected:
        std::vector<typename ElasticProblemSup<dim>::BoundaryValues> boundary_values;
//        BoundaryValues<dim>   boundary_values;
        // dealii::FESystem<dim> finite_element;
        std::string           output_file_name;
};

