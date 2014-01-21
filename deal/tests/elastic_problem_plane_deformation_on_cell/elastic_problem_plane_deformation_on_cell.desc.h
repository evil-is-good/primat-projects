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

#ifndef ELASTIC_PROBLEM_PLANE_DEFORMATION_ON_CELL_DESC

#define ELASTIC_PROBLEM_PLANE_DEFORMATION_ON_CELL_DESC

#include <projects/deal/tests/esm_elastic_problem/esm_elastic_problem.h>
#include <projects/deal/tests/erhsv_elastic_problem_on_cell/erhsv_elastic_problem_on_cell.h>
#include <projects/deal/main/problem/problem.h>
#include <projects/deal/main/solver_se/solver_se.h>
#include <projects/deal/main/function/function.h>
#include <projects/deal/main/domain_looper/sources/domain_looper.h>

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

#include <array>
#include <fstream>
#include <iostream>
#include </usr/local/include/boost/lexical_cast.hpp>

template <uint8_t dim>
class ElasticProblemPlaneDeformationOnCell : public Problem< 
            dim, 
            ElementStiffnessMatrixElasticProblem<dim>, 
            ElementRightHandSideVectorElasicProblemOnCell<dim> >
{
    public:
        ElasticProblemPlaneDeformationOnCell (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim>::TypeCoef &coef);

        ~ElasticProblemPlaneDeformationOnCell ();

        friend class Elasticproblem2DOnCell;

    //Methods
    public:
        virtual Report solved ();
        virtual void print_result (const std::string &filename);

    protected:
        virtual Report setup_system ();
        virtual Report assemble_matrix_of_system ();
        virtual Report assemble_right_vector_of_system ();
        virtual Report calculate_mean_coefficients ();
        virtual Report solve_system_equations ();
        virtual Report calculate_meta_coefficients ();
        virtual Report output_results ();

    //Fields
    public:
        double meta_coefficient[dim][dim][dim][dim];
        double mean_coefficient[dim][dim][dim][dim];
        
    protected:
        typename ElasticProblemSup<dim>::TypeCoef coefficient;

        dealii::Vector<double> solution[dim][dim];
        dealii::Vector<double> stress[dim][dim];
        
        BlackOnWhiteSubstituter black_on_white_substituter;
        
        dealii::FESystem<dim>  finite_element;
        std::string            output_file_name;
        
        double area_of_domain;
        std::vector<double> area_of_material;
};

#endif
