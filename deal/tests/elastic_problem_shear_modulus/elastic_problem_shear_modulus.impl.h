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

//#include "./elastic_problem_shear_modulus.desc.h"

template <int dim>
BoundaryValues<dim>::BoundaryValues ()
    :
        dealii::Function<dim> (dim)
{}

template <int dim>
void BoundaryValues<dim>::vector_value (const dealii::Point<dim> &p,
                                        dealii::Vector<double>   &values) const
{
    std::array<double, dim> res = this->content(p);

    for (size_t i = 0; i < dim; ++i)
        values(i) = res[i];
};

template <int dim>
void BoundaryValues<dim>::set (typename ElasticProblemSup<dim>::TypeFunc &value)
{
    content = value;
};

template<uint8_t dim>
ElementRightHandSideVectorElasticProblem<dim>::
ElementRightHandSideVectorElasticProblem ()
    :
        ElementRightHandSideVector<dim, double, typename ElasticProblemSup<dim>::TypeFunc> ()
{

};

template<uint8_t dim>
void ElementRightHandSideVectorElasticProblem<dim> :: 
set_coefficient (typename ElasticProblemSup<dim>::TypeFunc &coef)
{
    this->coefficient = coef;
};

template<uint8_t dim>
double ElementRightHandSideVectorElasticProblem<dim> :: 
operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0;

    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
        res += 
            (fe_values.shape_value (index_i, q_point) *
            this->coefficient(fe_values.quadrature_point(q_point))[index_i % dim] *
            fe_values.JxW(q_point));

    return res;
};

template<uint8_t dim>
ElasticProblem<dim>::ElasticProblem (
        const dealii::Triangulation<dim> &triangulation,
        typename ElasticProblemSup<dim>::TypeCoef &coefficient, 
        typename ElasticProblemSup<dim>::TypeFunc &boundary_values,
        typename ElasticProblemSup<dim>::TypeFunc &rhs_values) 
:
    Problem< 
        dim,
        ElementStiffnessMatrixElasticProblem<dim>,
        ElementRightHandSideVectorElasticProblem<dim> > (),
    finite_element (dealii::FE_Q<dim>(1), dim)
{
    this->domain.triangulation .copy_triangulation (triangulation);

    this->element_stiffness_matrix .set_coefficient (coefficient);

    this->boundary_values .set (boundary_values);

    this->element_rh_vector .set_coefficient (rhs_values);
};

template<uint8_t dim>
ElasticProblem<dim>::~ElasticProblem ()
{
    this->domain .clean ();
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::solved ()
{
    printf("1\n");
    REPORT setup_system ();
    printf("2\n");
    REPORT assemble_system ();
    printf("3\n");
    REPORT apply_boundary_values ();
    printf("4\n");
    REPORT solve_system_equations ();

    REPORT_USE( 
            Report report;
            report.result = _report .result;
            _return (report););
};

template<uint8_t dim>
void ElasticProblemShearModulus<dim>::print_result (const std::string &file_name)
{
    output_file_name = file_name;
    output_results ();
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    c_sparsity.compress ();

    this->system_equations .reinit (c_sparsity, 
            this->domain.dof_handler .n_dofs ());

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::assemble_system ()
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

        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
            area_of_domain += fe_values.JxW(q_point);
    };
//    for (size_t i = 0; i < this->system_equations.A.m(); ++i)
//        for (size_t j = 0; j < this->system_equations.A.m(); ++j)
//            if (this->system_equations.A.el(i,j))
//                printf("A[%ld][%ld]=%f\n", i,j,this->system_equations.A.el(i,j));
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::apply_boundary_values ()
{
    std::map<unsigned int,double> list_boundary_values;

    dealii::VectorTools::interpolate_boundary_values (
            this->domain.dof_handler,
            0,
            //dealii::ZeroFunction<dim>(dim),//
            boundary_values,
            list_boundary_values);

    printf("3.1\n");

    dealii::MatrixTools::apply_boundary_values (
            list_boundary_values,
            this->system_equations.A,
            this->system_equations.x,
            this->system_equations.b);

    printf("3.2\n");
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::solve_system_equations ()
{
    dealii::SolverControl solver_control (10000, 1e-12);

    Femenist::SolverSE<> solver(solver_control);

    REPORT solver .solve (
            this->system_equations.A,
            this->system_equations.x,
            this->system_equations.b);
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>::solve_system_equations ()
{
    size_t len_vector_solution = this->domain.dof_handler.n_dofs();

    double temp = 0.0;

    for (size_t i = 0; i < len_vector_solution; ++i)
    {
        temp += 
            this->system_equations.x[i] *
            -this->system_equations.b[i];
    };

    meta_shear_modulus = temp / area_of_domain;

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemShearModulus<dim>:: output_results ()
{
    for(size_t i = 0; i < this->domain.dof_handler.n_dofs(); ++i)
        printf("x[%ld]=%f \n", i, this->system_equations.x(i));

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (this->domain.dof_handler);
    
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
     };

    data_out.add_data_vector (this->system_equations.x, solution_names);
    data_out.build_patches ();

    std::ofstream output (output_file_name.data());
    data_out.write_gnuplot (output);

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

#endif









