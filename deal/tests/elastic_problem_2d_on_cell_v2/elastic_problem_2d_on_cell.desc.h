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
//#include <projects/deal/main/solver_se/solver_se.h>
#include <projects/deal/main/function/function.h>
#include <projects/deal/main/domain_looper/sources/domain_looper.h>

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <array>
#include <fstream>
#include <iostream>
#include </usr/local/include/boost/lexical_cast.hpp>

#include "projects/calculation_core/blocks/general/laplacian/vector/laplacian_vector.h"


template <uint8_t dim>
class ElasticProblem2DOnCellV2 : public Problem< 
            dim, 
            ElementStiffnessMatrixElasticProblem<dim>, 
            ElementRightHandSideVectorElasicProblemOnCell<dim> >
{
//    public:
//        static const uint8_t num_solutions = (dim * (dim + 1)) / 2;

    protected:
        enum coor {x, y, z};
        enum item_problem {xx, xy, xz, yx, yy, yz, zx, zy, zz};
        enum {Solution, Stress};

    public:
        ElasticProblem2DOnCellV2 (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim + 1>::TypeCoef &coef);

        ~ElasticProblem2DOnCellV2 ();

    //Methods
    public:
        virtual prmt::Report solved ();
        virtual void print_result (const std::string &filename);

    protected:
        virtual prmt::Report setup_system ();
        virtual prmt::Report assemble_matrix_of_system ();
        virtual prmt::Report assemble_right_vector_of_system ();
        virtual prmt::Report calculate_mean_coefficients ();
        virtual prmt::Report solve_system_equations ();
        virtual prmt::Report calculate_meta_coefficients ();
        virtual prmt::Report calculate_stress_tau ();
        virtual prmt::Report output_results ();

        prmt::Report assemble_right_vector_of_system_parallel (
                typename dealii::Vector<double>& b);
        prmt::Report solve_system_equations_parallel (
                typename dealii::Vector<double>& x, 
                typename dealii::Vector<double>& b);

        typename HeatConductionProblemSup<dim>::TypeCoef coef_for_problem_of_torsion_rod
            (const typename ElasticProblemSup<dim + 1>::TypeCoef &coef) const;

        template<uint8_t type_res>
        double human (uint8_t theta, uint8_t lambda, size_t index) const;

    //Fields
    public:
        double meta_coefficient[dim + 1][dim + 1][dim + 1][dim + 1];
        double mean_coefficient[dim + 1][dim + 1][dim + 1][dim + 1];
        HeatConductionProblemOnCell<dim> problem_of_torsion_rod;
        dealii::Vector<double> solution[dim + 1][dim + 1];
        vec<std::pair<dealii::Point<2>, arr<arr<arr<arr<dbl,3>,3>,3>,3> >> stress_tau;
        
    protected:
        typename ElasticProblemSup<dim + 1>::TypeCoef coefficient;

        dealii::Vector<double> stress[dim + 1][dim + 1];
//        std::vector<double> solution1[dim + 1][dim + 1];
//        std::vector<double> stress1[dim + 1][dim + 1];
        
        BlackOnWhiteSubstituter black_on_white_substituter;
        
        dealii::FESystem<dim>  finite_element;
        std::string            output_file_name;
        
        double area_of_domain;
        std::vector<double> area_of_material;

};

#endif
