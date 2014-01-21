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

#ifndef ELASTIC_PROBLEM_2D_ON_CELL_V2_DESC

#define ELASTIC_PROBLEM_2D_ON_CELL_V2_DESC

#include <projects/deal/tests/esm_elastic_problem/esm_elastic_problem.h>
#include <projects/deal/tests/erhsv_elastic_problem_on_cell/erhsv_elastic_problem_on_cell.h>
#include <projects/deal/tests/heat_conduction_problem_on_cell/heat_conduction_problem_on_cell.h>
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
class ElasticProblem2DOnCellV2 : public Problem< 
            dim, 
            ElementStiffnessMatrixElasticProblem<dim>, 
            ElementRightHandSideVectorElasicProblemOnCell<dim> >
{
//    public:
//        static const uint8_t num_solutions = (dim * (dim + 1)) / 2;

    public:
        ElasticProblem2DOnCellV2 (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim + 1>::TypeCoef &coef);

        ~ElasticProblem2DOnCellV2 ();

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

        typename HeatConductionProblemSup<dim>::TypeCoef coef_for_problem_of_torsion_rod
            (const typename ElasticProblemSup<dim + 1>::TypeCoef &coef) const;

    //Fields
    public:
        double meta_coefficient[dim + 1][dim + 1][dim + 1][dim + 1];
        double mean_coefficient[dim + 1][dim + 1][dim + 1][dim + 1];
        
    protected:
        typename ElasticProblemSup<dim + 1>::TypeCoef coefficient;

        dealii::Vector<double> solution[dim][dim];
        dealii::Vector<double> stress[dim][dim];
        
        BlackOnWhiteSubstituter black_on_white_substituter;
        
        dealii::FESystem<dim>  finite_element;
        std::string            output_file_name;
        
        double area_of_domain;
        std::vector<double> area_of_material;

        HeatConductionProblemOnCell<dim> problem_of_torsion_rod;
};

#endif
