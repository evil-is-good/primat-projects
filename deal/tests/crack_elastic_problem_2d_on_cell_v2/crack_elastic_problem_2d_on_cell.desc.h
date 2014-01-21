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
//#include <projects/deal/main/function/function.h>
#include <projects/deal/main/domain_looper/sources/domain_looper.h>
#include <projects/cae/main/domain_looper/domain_looper.h>
#include <projects/cae/main/loop_condition/loop_condition.h>

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
        static cu8 num_cources = 3; // x, y, z
        struct ValuesProblems {arr<arr<dbl, dim + 1>, dim + 1> problem;};

    public:
        ElasticProblem2DOnCellV2 (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim + 1>::TypeCoef &coef,
                const vec<prmt::LoopCondition<dim>> &loop_border);

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
        virtual Report calculate_cells_stress ();
        virtual Report calculate_cells_area ();
        virtual Report output_results ();
        
        Report calculate_main_stress ();
        std::array<double, 3> calculate_phisical_meta ();
        dbl compute_max_complete_main_stress_1 (vec<st> &cells);
        Report compute_borders ();
        bool at_boundary (vec<st> &cells);
        Report miscarry_broken_cells (vec<st> &cells);
        Report print_stress (cst indx_i, cst indx_j);

        Report assemble_right_vector_of_system_parallel (
                typename dealii::Vector<double>& b);
        Report solve_system_equations_parallel (
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
        vec<std::pair<dealii::Point<2>, 
            arr<arr<ValuesProblems, num_cources>, num_cources>>> cells_stress;
        vec<dbl> complete_main_stress_1;
        vec<dbl> complete_main_stress_2;
        vec<arr<arr<dbl, num_cources>, num_cources>> complete_stress;
        vec<dbl> max_complete_main_stress_1;
        vec<vec<vec<st>>> brocken_cell;
        vec<st> brocken_cell_in_line; 
        vec<dealii::Point<dim>> brocken_cell_in_line_coor;
        vec<dbl> cells_area;
        vec<int> material_of_cell;
        vec<arr<dbl, 3>> fiz_coef;

        dealii::FESystem<dim>  finite_element;
        
    protected:
        typename ElasticProblemSup<dim + 1>::TypeCoef coefficient;

        dealii::Vector<double> stress[dim + 1][dim + 1];

        struct Border
        {
            double coor[dim][2]; // 0 - black, 1 - white;

            inline double* operator[] (int i)
            {
                return coor[i];
            };
        } border;
//        std::vector<double> solution1[dim + 1][dim + 1];
//        std::vector<double> stress1[dim + 1][dim + 1];
        
        typename prmt::BlackOnWhiteSubstituter black_on_white_substituter;
        // BlackOnWhiteSubstituter black_on_white_substituter;
        
        // dealii::FESystem<dim>  finite_element;
        std::string            output_file_name;
        
        double area_of_domain;
        std::vector<double> area_of_material;

        vec<prmt::LoopCondition<dim>> loop_condition;

};

#endif
