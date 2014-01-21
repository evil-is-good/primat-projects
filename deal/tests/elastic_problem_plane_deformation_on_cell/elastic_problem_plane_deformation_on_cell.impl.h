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

#ifndef ELASTIC_PROBLEM_PLANE_DEFORMATION_ON_CELL_IMPL

#define ELASTIC_PROBLEM_PLANE_DEFORMATION_ON_CELL_IMPL

//#include "./elastic_problem_plane_deformation_on_cell.desc.h"

template<uint8_t dim>
ElasticProblemPlaneDeformationOnCell<dim>::
ElasticProblemPlaneDeformationOnCell (
        const dealii::Triangulation<dim> &triangulation,
        const typename ElasticProblemSup<dim>::TypeCoef &coef)
:
    Problem< 
        dim,
        ElementStiffnessMatrixElasticProblem<dim>,
        ElementRightHandSideVectorElasicProblemOnCell<dim> > (),
    finite_element (dealii::FE_Q<dim>(1), dim)
{
    this->element_stiffness_matrix .set_coefficient (coef);

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                {
                    this->coefficient[i][j][k][l] .clear ();

                    for (size_t m = 0; m < coef[i].size(); ++m)
                        this->coefficient[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
                };

    this->domain.triangulation .copy_triangulation (triangulation);
};

template<uint8_t dim>
ElasticProblemPlaneDeformationOnCell<dim>::
~ElasticProblemPlaneDeformationOnCell ()
{
    this->domain .clean ();
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::solved ()
{
    printf("AAA 1\n");
    REPORT setup_system ();
    printf("AAA 2\n");
    REPORT assemble_matrix_of_system ();
    printf("AAA 3\n");
    REPORT calculate_mean_coefficients ();
//    printf("%f, %f, %f\n", mean_coefficient[0],
//                           mean_coefficient[1],
//                           mean_coefficient[2]);
    printf("AAA 4\n");

    std::array<std::array<std::vector<double>, dim>, dim> coef_for_rhs;
    for(size_t i = 0; i < dim; ++i)
        for(size_t j = 0; j < dim; ++j)
            coef_for_rhs[i][j] .resize (coefficient[i][j][0][0].size());
    
    for(size_t theta = 0; theta < dim; ++theta)
        for(size_t lambda = theta; lambda < dim; ++lambda)
        {
            for(size_t i = 0; i < dim; ++i)
                for(size_t j = 0; j < dim; ++j)
                    for(size_t k = 0; 
                            k < coefficient[i][j][theta][lambda].size(); ++k)
                    {
                        coef_for_rhs[i][j][k] = 
                            coefficient[i][j][theta][lambda][k];
                        printf("theta=%ld lambda=%ld %f\n", theta, lambda,
                                coefficient[i][j][theta][lambda][k]);
                    };
        
            this->system_equations.x = 0;
            this->system_equations.b = 0;

            this->element_rh_vector .set_coefficient (coef_for_rhs);

            REPORT assemble_right_vector_of_system ();

            REPORT solve_system_equations ();

            for (size_t i = 0; i < this->system_equations.x.size(); ++i)
                this->system_equations.x(i) = this->system_equations.x(
                        black_on_white_substituter .subst (i));
            //            if (black_on_white_substituter .is_black (i))
            //                this->system_equations.x(i) = 0.0;

            solution[theta][lambda] = this->system_equations.x;

            stress[theta][lambda] = this->system_equations.b;
        };

    printf("AAA 5\n");
    REPORT calculate_meta_coefficients ();
    printf("AAA 6\n");

    REPORT_USE( 
            Report report;
            report.result = _report .result;
            _return (report););
};

template<uint8_t dim>
void ElasticProblemPlaneDeformationOnCell<dim>::
print_result (const std::string &file_name)
{
    output_file_name = file_name;
    output_results ();
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    std::ofstream output1 ("csp.1");
    c_sparsity .print_gnuplot (output1);

    DomainLooper<dim, true> dl;
    REPORT dl .loop_domain(
            this->domain.dof_handler,
            black_on_white_substituter,
            c_sparsity);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    std::ofstream output2 ("csp.2");
    c_sparsity .print_gnuplot (output2);

    c_sparsity.compress ();

    this->system_equations .reinit (c_sparsity, 
            this->domain.dof_handler.n_dofs());
    printf("??????????\n");
    
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = i; j < dim; ++j)
        {
            solution[i][j] .reinit (this->domain.dof_handler.n_dofs());
            stress[i][j]   .reinit (this->domain.dof_handler.n_dofs());
        };
    printf("AAAAAAAAAAAAAAAAA\n");

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::assemble_matrix_of_system ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

    dealii::FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);
        cell_matrix = 0;

        for (size_t i = 0; i < dofs_per_cell; ++i)
            for (size_t j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i,j) += this->element_stiffness_matrix
                    (i, j, quadrature_formula, fe_values, cell->material_id()); 

        cell ->get_dof_indices (local_dof_indices);

//        for (size_t i = 0; i < dofs_per_cell; ++i)
//            printf("loc1 %d\n", local_dof_indices[i]);

        for (size_t i = 0; i < dofs_per_cell; ++i)
            local_dof_indices[i] = black_on_white_substituter .subst (
                    local_dof_indices[i]);

//        for (size_t i = 0; i < dofs_per_cell; ++i)
//            printf("loc2 %d\n", local_dof_indices[i]);

        for (size_t i = 0; i < dofs_per_cell; ++i)
            for (size_t j = 0; j < dofs_per_cell; ++j)
               this->system_equations.A .add (local_dof_indices[i],
                                              local_dof_indices[j],
                                              cell_matrix(i,j));
        printf("AAAAAAAAAAAAAAA %f\n", this->system_equations.A.el(0,0));
    };
    for (size_t i = 0; i < this->system_equations.A.m(); ++i)
        for (size_t j = 0; j < this->system_equations.A.m(); ++j)
            if (this->system_equations.A.el(i,j))
                printf("A[%ld][%ld]=%f\n", i,j,this->system_equations.A.el(i,j));
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::
assemble_right_vector_of_system ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

    dealii::Vector<double> cell_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);
        cell_rhs = 0;

        for (size_t i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += this->element_rh_vector
                    (i, quadrature_formula, fe_values, cell->material_id()); 

        cell ->get_dof_indices (local_dof_indices);

        for (size_t i = 0; i < dofs_per_cell; ++i)
            local_dof_indices[i] = black_on_white_substituter .subst (
                    local_dof_indices[i]);

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
//            if (local_dof_indices[i] == 3)
//                printf("%f\n", cell_rhs(i));
            this->system_equations.b (local_dof_indices[i]) +=
                cell_rhs (i);
        };
    };
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::calculate_mean_coefficients ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    mean_coefficient[i][j][k][l] = 0.0;

    area_of_domain = 0.0;

    area_of_material .resize (this->coefficient[0][0][0][0].size());

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);

//        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
//            for (size_t i = 0; i < dim; ++i)
//                for (size_t j = 0; j < dim; ++j)
//                    for (size_t k = 0; k < dim; ++k)
//                        for (size_t l = 0; l < dim; ++l)
//                            mean_coefficient[i][j][k][l] += 
//                                this->coefficient[i][j][k][l][cell->material_id()] *
//                                fe_values.JxW(q_point);
//
//        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
//            area_of_domain += fe_values.JxW(q_point);

        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
            area_of_material[cell->material_id()] += fe_values.JxW(q_point);
    };

    for (size_t i = 0; i < area_of_material.size(); ++i)
        area_of_domain += area_of_material[i];

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                {
                    for (size_t m = 0; m < area_of_material.size(); ++m)
                    mean_coefficient[i][j][k][l] +=
                        this->coefficient[i][j][k][l][m] * 
                        area_of_material[m];

                    mean_coefficient[i][j][k][l] /= area_of_domain;
                };

//    printf("AREA=%f\n", area_of_domain);
//
//    area_of_domain /= 2;
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::calculate_meta_coefficients ()
{
    size_t len_vector_solution = this->domain.dof_handler.n_dofs();
    double mean_stress[dim][dim][dim][dim];

//    printf("META!!!!!!!!!!!!!!!!!\n");
//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = 0; j < dim; ++j)
//            printf("%ld %ld %ld\n", i, j, conver(i,j));
//    printf("sol\n");
//    for (size_t i = 0; i < dim; ++i)
//    {
//        for (size_t k = 0; k < len_vector_solution; ++k)
//            printf("%f\n", solution[i](k));
//        printf("\n");
//    };
//    printf("flow\n");
//    for (size_t i = 0; i < dim; ++i)
//    {
//        for (size_t k = 0; k < len_vector_solution; ++k)
//            printf("%f\n", heat_flow[i](k));
//        printf("\n");
//    };

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    meta_coefficient[i][j][k][l] = 
                        mean_coefficient[i][j][k][l];

//    reducing to 2d view

    uint8_t width_2d_matrix = dim * dim;

    for (size_t i = 0; i < width_2d_matrix; ++i)
        for (size_t j = i; j < width_2d_matrix; ++j)
        {
            uint8_t im = i % dim;
            uint8_t in = i / dim;

            uint8_t jm = j % dim;
            uint8_t jn = j / dim;

            printf("%d %d %d %d\n", im, in, jm, jn);

            mean_stress[im][in][jm][jn] = 0.0;

            uint8_t a = im;
            uint8_t b = in;
            uint8_t c = jm;
            uint8_t d = jn;

            if (im == 1)
                a ^= b ^= a ^= b;

            if (jm == 1)
                c ^= d ^= c ^= d;

            for (size_t k = 0; k < len_vector_solution; ++k)
                mean_stress[im][in][jm][jn] += 
                    solution[a][b](k) * (-stress[c][d](k));

            printf("STRESS=%f\n", mean_stress[im][in][jm][jn]);

            mean_stress[im][in][jm][jn] /= area_of_domain; 

            meta_coefficient[im][in][jm][jn] += 
                mean_stress[im][in][jm][jn];

            meta_coefficient[jm][jn][im][in] = 
                meta_coefficient[im][in][jm][jn];
        };

//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = i; j < dim; ++j)
//            for (size_t k = 0; k < dim; ++k)
//                for (size_t l = k; l < dim; ++l)
//                {
//                    mean_stress[i][j][k][l] = 0.0;
//
//                    for (size_t m = 0; m < len_vector_solution; ++m)
//                        mean_stress[i][j][k][l] += 
//                            solution[i][j](m) * (-stress[k][l](m));
//
//                    mean_stress[i][j][k][l] /= area_of_domain; 
//
//                    meta_coefficient[i][j][k][l] += 
//                        mean_stress[i][j][k][l];
//                    printf("%ld %ld %ld %ld\n", i, j, k, l);
//                };
   
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblemPlaneDeformationOnCell<dim>::solve_system_equations ()
{
    dealii::SolverControl solver_control (10000, 1e-12);

    Femenist::SolverSE<> solver(solver_control);

    this->system_equations.x = 0;

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
Report ElasticProblemPlaneDeformationOnCell<dim>:: output_results ()
{
    for (uint8_t i = 0; i < dim; ++i)
        for (uint8_t j = i; j < dim; ++j)
        {
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler (this->domain.dof_handler);

            char suffix[3] = {'x', 'y', 'z'};
    
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

            data_out.add_data_vector (solution[i][j], solution_names);
            data_out.build_patches ();

            std::string file_name = output_file_name;
            file_name += suffix[i];
            file_name += suffix[j];
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

#endif









