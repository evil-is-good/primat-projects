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

#ifndef HEAT_CONDUCTION_PROBLEM_DESC

#define HEAT_CONDUCTION_PROBLEM_DESC

#include <array>
#include <projects/deal/tests/esm_laplace_problem/esm_laplace_problem.h>
#include <projects/deal/main/problem/problem.h>
#include <projects/deal/main/solver_se/solver_se.h>
#include <projects/deal/main/function/function.h>
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
#include <deal.II/lac/precondition.h>
#include <fstream>
#include <iostream>

//template <int dim>
//class BoundaryValues : public dealii::Function<dim>
//{
//  public:
//    BoundaryValues () : dealii::Function<dim>() {}
//
//    virtual double value (const dealii::Point<dim>   &p,
//                          const unsigned int  component = 0) const;
//
//    void set (Femenist::Function<double, dim> &value);
//
//    Femenist::Function<double, dim> content;
//};

namespace Femenist
{
    template <int dim>
        class MyFuncFromDealii : public dealii::Function<dim>
    {
        public:
            typedef std::function<double (const dealii::Point<dim>&)> Func;

        public:
            MyFuncFromDealii () : dealii::Function<dim>() {};

            MyFuncFromDealii (Func &func)
                : dealii::Function<dim>() 
            {
                this->func = func;
            };

            MyFuncFromDealii (const MyFuncFromDealii& mffd)
            {
                this->func = mffd.func;
            };

            void operator= (Func &func)
            {
                this->func = func;
            };

            virtual double value (const dealii::Point<dim>   &p,
                    const unsigned int  component = 0) const
            {
                return func (p);
            };

        private:
            Func func;
    };

    template<uint8_t dim>
        struct BoundaryValues
        {
            MyFuncFromDealii<dim> function;
            size_t boundari_indicator;
            uint8_t type;

            static const uint8_t Dirichlet = 0;
            static const uint8_t Neumann   = 1;
        };
};

template<uint8_t dim>
class ElementRightHandSideVectorHeatConductionProblem : 
    public ElementRightHandSideVector<dim, double, Femenist::Function<double, dim>>
{
    public:
        ElementRightHandSideVectorHeatConductionProblem ();

    virtual void set_coefficient (Femenist::Function<double, dim> &coef); 

    virtual double operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values) const;
};

template <uint8_t dim>
class HeatConductionProblem : public Problem< 
                              dim, 
                              ElementStiffnessMatrixLaplaceProblem<dim>, 
                              ElementRightHandSideVectorHeatConductionProblem<dim> >
{
//    public:
//        uint8_t static const num_coef = (dim * (dim + 1)) / 2;

    public:
        HeatConductionProblem (const dealii::Triangulation<dim> &triangulation,
                               typename HeatConductionProblemSup<dim>::TypeCoef &coefficient, 
                               std::vector<Femenist::BoundaryValues<dim> > &boundary_values,
                               typename Femenist::Function<double, dim> &rhs_values);
        ~HeatConductionProblem ();
    //Methods
    public:
        virtual prmt::Report solved ();
        virtual void print_result (const std::string &filename);

    protected:
        virtual prmt::Report setup_system ();
        virtual prmt::Report assemble_system ();
        virtual prmt::Report apply_boundary_values ();
        virtual prmt::Report solve_system_equations ();
        virtual prmt::Report output_results ();

    //Fields
    protected:
        std::vector<Femenist::BoundaryValues<dim> > boundary_values;
        dealii::FE_Q<dim> finite_element;
        std::string output_file_name;
};

#endif
