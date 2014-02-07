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

#ifndef HEAT_CONDUCTION_PROBLEM_ON_CELL_DESC

#define HEAT_CONDUCTION_PROBLEM_ON_CELL_DESC

#include <projects/deal/tests/esm_laplace_problem/esm_laplace_problem.h>
#include <projects/deal/main/element_stiffness_matrix/element_stiffness_matrix.desc.h>
#include <projects/deal/main/problem/problem.h>
#include <projects/deal/main/solver_se/solver_se.h>
#include <projects/deal/main/function/function.h>
#include <projects/deal/main/domain_looper/sources/domain_looper.h>

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
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
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/relaxation_block.h>
#include <omp.h>

#include <array>
#include <fstream>
#include <iostream>
// #include </usr/local/include/boost/lexical_cast.hpp>

#include <time.h>

template<uint8_t dim>
class ElementRightHandSideVectorHeatConductionProblemOnCell : 
    public ElementRightHandSideVector<
        dim, 
        double,
        std::array<std::vector<double>, dim> >
//        std::array<Femenist::Function<double,dim>,dim> >
{
    public:
        ElementRightHandSideVectorHeatConductionProblemOnCell ();

    virtual void set_coefficient (
            const std::array<std::vector<double>, dim> &coef);
//            std::array<Femenist::Function<double,dim>,dim> &coef); 

    virtual double operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const;
};

template <uint8_t dim>
class HeatConductionProblemOnCell : public Problem< 
            dim, 
            ElementStiffnessMatrixLaplaceProblem<dim>, 
            ElementRightHandSideVectorHeatConductionProblemOnCell<dim> >
{
    public:
        uint8_t static const num_coef = (dim * (dim + 1)) / 2;

//        typedef std::array<
//                Femenist::Function<double, dim>, num_coef> 
        typedef std::array<std::vector<double>, num_coef> ContainerCoef;

    public:
        HeatConductionProblemOnCell (const dealii::Triangulation<dim> &triangulation,
                                     const ContainerCoef &coef);
        ~HeatConductionProblemOnCell ();

    //Methods
    public:
        virtual prmt::Report solved ();
        virtual void print_result (const std::string &filename);
        virtual uint8_t conver (uint8_t index_i, uint8_t index_j);

    // protected:
    public:
        virtual prmt::Report setup_system ();
        virtual prmt::Report assemble_matrix_of_system ();
        virtual prmt::Report assemble_right_vector_of_system ();
        virtual prmt::Report calculate_mean_coefficients ();
        virtual prmt::Report solve_system_equations ();
        virtual prmt::Report calculate_meta_coefficients ();
        virtual prmt::Report calculate_flow ();
        virtual prmt::Report output_results ();
        virtual prmt::Report calculate_cells_area ();

        prmt::Report assemble_right_vector_of_system_parallel (
                typename dealii::Vector<double>& b);
        prmt::Report solve_system_equations_parallel (
                typename dealii::Vector<double>& x, 
                typename dealii::Vector<double>& b);


    //Fields
    public:
        double meta_coefficient[num_coef];
        double mean_coefficient[num_coef];
        dealii::Vector<double> solution[dim];
        dealii::Vector<double> heat_flow[dim];
        dealii::Vector<double> anal_x; ///////////
        vec<std::pair<dealii::Point<2>, arr<arr<double, dim>, dim > > > flow;
        vec<dbl> cells_area;
        ContainerCoef coefficient;
        vec<st> brocken_cell_in_line; 
        
    // protected:
    public:
        BlackOnWhiteSubstituter black_on_white_substituter;
        dealii::FE_Q<dim>  finite_element;
        std::string output_file_name;
        double area_of_domain;
};

#endif
