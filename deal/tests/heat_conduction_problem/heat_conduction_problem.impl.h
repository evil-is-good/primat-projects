/*
 * =====================================================================================
 *
 *       Filename:  heat_conduction_problem.impl.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  14.09.2012 12:20:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef HEAT_CONDUCTION_PROBLEM_IMPL

#define HEAT_CONDUCTION_PROBLEM_IMPL

//#include "./heat_conduction_problem.desc.h"

//template <int dim>
//double BoundaryValues<dim>::value (const dealii::Point<dim> &p,
//                                   const unsigned int /*component*/) const
//{
//    return this->content(p);
//};
//
//template <int dim>
//void BoundaryValues<dim>::set (Femenist::Function<double, dim> &value)
//{
//    content = value;
//};


template<uint8_t dim>
ElementRightHandSideVectorHeatConductionProblem<dim>::
ElementRightHandSideVectorHeatConductionProblem ()
    :
        ElementRightHandSideVector<dim, double, Femenist::Function<double, dim>> ()
{

};

template<uint8_t dim>
void ElementRightHandSideVectorHeatConductionProblem<dim> :: 
set_coefficient (Femenist::Function<double, dim> &coef)
{
    this->coefficient = coef;
};

template<uint8_t dim>
double ElementRightHandSideVectorHeatConductionProblem<dim> :: 
operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0;

    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
        res += 
            (fe_values.shape_value (index_i, q_point) *
            this->coefficient(fe_values.quadrature_point(q_point)) *
            fe_values.JxW(q_point));

    return res;
};


template<uint8_t dim>
HeatConductionProblem<dim>::HeatConductionProblem (
        const dealii::Triangulation<dim> &triangulation,
        typename HeatConductionProblemSup<dim>::TypeCoef &coefficient, 
        std::vector<Femenist::BoundaryValues<dim> > &boundary_values,
        typename Femenist::Function<double, dim> &rhs_values)
:
    Problem< 
        dim,
        ElementStiffnessMatrixLaplaceProblem<dim>,
        ElementRightHandSideVectorHeatConductionProblem<dim> > (),
    finite_element (1)
{
    this->domain.triangulation .copy_triangulation (triangulation);

    this->element_stiffness_matrix .set_coefficient (coefficient);

//    this->boundary_values .set (boundary_values);
//    this->boundary_values .resize (boundary_values.size());
//    std::copy(boundary_values.begin(), boundary_values.end(),
//            this->boundary_values.begin());
    for (Femenist::BoundaryValues<dim> bv : boundary_values)
        this->boundary_values .push_back (bv);

    this->element_rh_vector .set_coefficient (rhs_values);
};

template<uint8_t dim>
HeatConductionProblem<dim>::~HeatConductionProblem ()
{
    this->domain .clean ();
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>::solved ()
{
    REPORT setup_system ();
    REPORT assemble_system ();
    {
        FILE *F;
        F = fopen("matrix.gpd","w");
        for (size_t i = 0; i < this->system_equations.A.m(); ++i)
            for (size_t j = 0; j < this->system_equations.A.n(); ++j)
                if (this->system_equations.A.el(i,j))
                    fprintf(F,"%ld %ld %f\n", i, j, this->system_equations.A.el(i,j));
        fclose(F);
    };
    REPORT apply_boundary_values ();
    {
        FILE *F;
        F = fopen("A.gpd","w");
        for (size_t i = 0; i < this->system_equations.A.m(); ++i)
        {
            for (size_t j = 0; j < this->system_equations.A.n()-1; ++j)
                fprintf(F,"%f ", this->system_equations.A .el (i,j));
            fprintf(F,"%f\n", this->system_equations.A.el(i,this->system_equations.A.n()-1));
        };
        fclose(F);
    };
    {
        FILE *F;
        F = fopen("b.gpd","w");
        for (size_t i = 0; i < this->system_equations.b.size(); ++i)
            fprintf(F,"%f\n", this->system_equations.b(i));
        fclose(F);
    };
    REPORT solve_system_equations ();
    {
        FILE *F;
        F = fopen("x.gpd","w");
        for (size_t i = 0; i < this->system_equations.x.size(); ++i)
            fprintf(F,"%f\n", this->system_equations.x(i));
        fclose(F);
    };

    REPORT_USE( 
            prmt::Report report;
            report.result = _report .result;
            _return (report););
};

template<uint8_t dim>
void HeatConductionProblem<dim>::print_result (const std::string &file_name)
{
    output_file_name = file_name;
    output_results ();
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    this->system_equations .reinit (c_sparsity, 
            this->domain.dof_handler .n_dofs ());

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>::assemble_system ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_values   | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int   dofs_per_cell = finite_element.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    dealii::FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell!=endc; ++cell)
    {
        fe_values .reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
            for (size_t j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i,j) += this->element_stiffness_matrix
                    (i, j, quadrature_formula, fe_values, cell->material_id()); 

            cell_rhs(i) += this->element_rh_vector 
                (i, quadrature_formula, fe_values); 
        };

        cell ->get_dof_indices (local_dof_indices);

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
            for (size_t j = 0; j < dofs_per_cell; ++j)
               this->system_equations.A  .add (local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i,j));

            this->system_equations.b (local_dof_indices[i]) += cell_rhs(i);
        };
    };
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>::apply_boundary_values ()
{
    for (Femenist::BoundaryValues<dim> bv : this->boundary_values)
    {
        if (bv.type _is Femenist::BoundaryValues<dim>::Dirichlet)
        {
            std::map<unsigned int,double> list_boundary_values;

            dealii::VectorTools::interpolate_boundary_values (
                    this->domain.dof_handler,
                    bv.boundari_indicator,
                    bv.function,
                    list_boundary_values);

            dealii::MatrixTools::apply_boundary_values (
                    list_boundary_values,
                    this->system_equations.A,
                    this->system_equations.x,
                    this->system_equations.b);
        }
        else if (bv.type _is Femenist::BoundaryValues<dim>::Neumann)
        {
            dealii::Vector<double> tmp (this->system_equations.b.size());

            dealii::VectorTools::create_boundary_right_hand_side (
                    this->domain.dof_handler,
                    dealii::QGauss<dim-1>(2),
                    bv.function,
                    tmp);

            this->system_equations.b += tmp;
        };
    };

    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>::solve_system_equations ()
{
    dealii::SolverControl solver_control (10000, 1e-12);

//    Femenist::SolverSE<> solver(solver_control);
//
//    REPORT solver .solve (
//            this->system_equations.A,
//            this->system_equations.x,
//            this->system_equations.b);

    for (size_t i = 0; i < this->system_equations.b.size(); ++i)
    {
//        temp +=
//                this->system_equations.A .el (22,i) *
//                anal_x(i);
//        printf("AAAA %ld %f %f %f %f\n", 
//                i,
//                this->system_equations.A .el (22,i),
//                anal_x(i),
//                this->system_equations.A .el (22,i) *
//                anal_x(i),
//                temp
//                );
//        if (not black_on_white_substituter.is_black(i))
            this->system_equations.x(i) = this->system_equations.A .el (13,i);
            printf("%ld %f\n", i, this->system_equations.A .el (13,i));
    }

    dealii::PreconditionSSOR<> preconditioner;
       preconditioner.initialize(this->system_equations.A, 1.2);

//    dealii::SolverCG<> solver (solver_control);
//    solver.solve (
//            this->system_equations.A,
//            this->system_equations.x,
//            this->system_equations.b
////            ,dealii::PreconditionIdentity()
//            ,preconditioner
//            );
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblem<dim>:: output_results ()
{
    {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (this->domain.dof_handler);
    data_out.add_data_vector (this->system_equations.x, "solution");
    data_out.build_patches ();

    auto name = output_file_name;
    name += "x.gpd";

    std::ofstream output (name);
    data_out.write_gnuplot (output);
    }

    {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (this->domain.dof_handler);
    data_out.add_data_vector (this->system_equations.b, "solution");
    data_out.build_patches ();

    auto name = output_file_name;
    name += "b.gpd";

    std::ofstream output (name);
    data_out.write_gnuplot (output);
    }

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

#endif









