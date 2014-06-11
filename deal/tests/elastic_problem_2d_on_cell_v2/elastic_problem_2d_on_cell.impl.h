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

#ifndef ELASTIC_PROBLEM_2D_ON_CELL_V2_IMPL

#define ELASTIC_PROBLEM_2D_ON_CELL_V2_IMPL

//#include "./elastic_problem_2d_on_cell.desc.h"

template <size_t begin, size_t end, size_t step = 1>
struct range : public std::array<size_t, ((begin - end) / step + 1)>
{
    range () 
    {
        size_t i = 0;
        for (size_t j = begin; j <= end; j += step, ++i)
            this->_M_instance[i] = j;
    };
};


template<uint8_t dim>
ElasticProblem2DOnCellV2<dim>::
ElasticProblem2DOnCellV2 (
        const dealii::Triangulation<dim> &triangulation,
        const typename ElasticProblemSup<dim + 1>::TypeCoef &coef)
:
    Problem< 
        dim,
        ElementStiffnessMatrixElasticProblem<dim>,
        ElementRightHandSideVectorElasicProblemOnCell<dim> > (),
    finite_element (dealii::FE_Q<dim>(1), dim),
    problem_of_torsion_rod (triangulation, coef_for_problem_of_torsion_rod (coef))
{
    typename ElasticProblemSup<dim>::TypeCoef coef_for_plane_deformation;

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    for (size_t m = 0; m < coef[i][j][k][l].size(); ++m)
                        coef_for_plane_deformation[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
    
    this->element_stiffness_matrix .set_coefficient (coef_for_plane_deformation);

    for (size_t i = 0; i < dim + 1; ++i)
        for (size_t j = 0; j < dim + 1; ++j)
            for (size_t k = 0; k < dim + 1; ++k)
                for (size_t l = 0; l < dim + 1; ++l)
                {
                    this->coefficient[i][j][k][l] .clear ();

                    for (size_t m = 0; m < coef[i][j][k][l].size(); ++m)
                        this->coefficient[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
                };

    this->domain.triangulation .copy_triangulation (triangulation);
};

template<uint8_t dim>
ElasticProblem2DOnCellV2<dim>::
~ElasticProblem2DOnCellV2 ()
{
    this->domain .clean ();
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::solved ()
{
    puts("11");
    REPORT setup_system ();
    puts("12");
    REPORT assemble_matrix_of_system ();
    puts("13");
    REPORT calculate_mean_coefficients ();
    puts("14");

    // 2d elastic problem 
    
    std::array<
        std::array<
        std::vector<double>,
        dim>,
        dim> coef_for_rhs;

    for (auto i : {0, 1})//range<0, 1>())
        for (auto j : {0, 1})//range<0, 1>())
            coef_for_rhs[i][j] .resize (coefficient[i][j][0][0].size());

//    for (auto theta : {x, y, z})
//        for (auto lambda : {x, y, z})
//            if ((theta == lambda) or ((theta == x) and (lambda == y)))
//            {
//                for (auto i : {x, y})
//                    for (auto j : {x, y})
//                        for(
//                                size_t k = 0; 
//                                k < coefficient[i][j][theta][lambda].size();
//                                ++k)
//                        {
//                            coef_for_rhs[i][j][k] = 
//                                coefficient[i][j][theta][lambda][k];
//                        };
//
//                this->system_equations.x = 0;
//                this->system_equations.b = 0;
//
//                this->element_rh_vector .set_coefficient (coef_for_rhs);
//
//                REPORT assemble_right_vector_of_system ();
//
//                REPORT solve_system_equations ();
//
//                for (size_t i = 0; i < this->system_equations.x.size(); ++i)
//                    this->system_equations.x(i) = this->system_equations.x(
//                            black_on_white_substituter .subst (i));
//
//                solution[theta][lambda] = this->system_equations.x;
//
//                stress[theta][lambda] = this->system_equations.b;
//            };

    for (auto theta : {x, y, z})
        for (auto lambda : {x, y, z})
            if ((theta == lambda) or ((theta == x) and (lambda == y)))
            {
                for (auto i : {x, y})
                    for (auto j : {x, y})
                        for(
                                size_t k = 0; 
                                k < coefficient[i][j][theta][lambda].size();
                                ++k)
                        {
                            coef_for_rhs[i][j][k] = 
                                coefficient[i][j][theta][lambda][k];
                        };

                solution[theta][lambda] = 0;
                stress[theta][lambda] = 0;
//                printf("%f\n", stress[theta][lambda](0));

                this->element_rh_vector .set_coefficient (coef_for_rhs);

                REPORT assemble_right_vector_of_system_parallel (
                        assigned_to stress[theta][lambda]);

                // FILE* F;
                // F = fopen("S.gpd", "a");
                // for (size_t i = 0; i < stress[theta][lambda].size(); ++i)
                //     fprintf(F, "%ld %f\n", i, stress[theta][lambda][i]);
                // fclose(F);
//                puts("11111111111d");
            };

//    for (auto theta : {x, y, z})
//        for (auto lambda : {x, y, z})
#pragma omp parallel for
    for (size_t theta = 0; theta < 3; ++theta)
        for (size_t lambda = 0; lambda < 3; ++lambda)
            if ((theta == lambda) or ((theta == x) and (lambda == y)))
            {
                REPORT solve_system_equations_parallel (
                        solution[theta][lambda], stress[theta][lambda]);

//                puts("11111111111d");
                for (size_t i = 0; i < this->system_equations.x.size(); ++i)
                    solution[theta][lambda](i) = solution[theta][lambda](
                            black_on_white_substituter .subst (i));
//                puts("sdfasdsdfsfdfgddgdfgdffgd");
            };

//                for (
//                        size_t i = 0; 
//                        i < this->system_equations.x.size();
//                        ++i
//                    )
//                {
////                    solution1[theta][lambda] .push_back(
////                            this->system_equations.x(i));
////                    stress1[theta][lambda] .push_back(
////                            this->system_equations.b(i));
//
//                    solution[theta][lambda](i) =
//                            this->system_equations.x(i);
//                    stress[theta][lambda](i) =
//                            this->system_equations.b(i);
//                };
//            };

//    static const uint8_t x = 0;
//    static const uint8_t y = 1;
//    static const uint8_t z = 2;

//    for(size_t i = 0; i < dim; ++i)
//        for(size_t j = 0; j < dim; ++j)
//            for(size_t k = 0; 
//                    k < coefficient[i][j][z][z].size(); ++k)
//            {
//                coef_for_rhs[i][j][k] = 
//                    coefficient[i][j][z][z][k];
//            };
//
//    this->system_equations.b = 0;
//
//    this->element_rh_vector .set_coefficient (coef_for_rhs);
//
//    REPORT assemble_right_vector_of_system ();

    // torsion rot problem

    problem_of_torsion_rod .solved ();

                FILE* F;
                F = fopen("Sol.gpd", "w");
                for (size_t i = 0; i < problem_of_torsion_rod.solution[0].size(); ++i)
                    fprintf(F, "%ld %.5f\n", i, problem_of_torsion_rod.solution[1][i]);
                fclose(F);

//    for (
//            size_t i = 0; 
//            i < problem_of_torsion_rod.solution[0].size();
//            ++i
//        )
//    {
//        solution1[x][z] .push_back (
//                problem_of_torsion_rod.solution[x](i));
//        solution1[y][z] .push_back (
//                problem_of_torsion_rod.solution[y](i));
//
//        stress1[x][z] .push_back (
//                problem_of_torsion_rod.heat_flow[x](i));
//        stress1[y][z] .push_back (
//                problem_of_torsion_rod.heat_flow[y](i));
//
//        for (auto theta : {x, y, z})
//            for (auto lambda : {x, y, z})
//                if ((theta == lambda) or ((theta == x) and (lambda == y)))
//                {
//                    solution1[theta][lambda] .push_back (0.0);
//                    stress1[theta][lambda]   .push_back (0.0);
//                };

//    for (
//            size_t i = this->system_equations.x.size(); 
//            i < (this->system_equations.x.size() + 
//                problem_of_torsion_rod.solution[0].size());
//            ++i
//        )
//    {
//        solution[x][z](i) = 
//                problem_of_torsion_rod.solution[x](i);
//        solution[y][z](i) = 
//                problem_of_torsion_rod.solution[y](i);
//
//        stress[x][z](i) =
//                problem_of_torsion_rod.heat_flow[x](i);
//        stress[y][z](i) = 
//                problem_of_torsion_rod.heat_flow[y](i);
//    };
//
//    // restore symmetry
//
//    solution[y][x] = solution[x][y];
//    solution[z][x] = solution[x][z];
//    solution[z][y] = solution[y][z];
//
//    stress[y][x] = stress[x][y];
//    stress[z][x] = stress[x][z];
//    stress[z][y] = stress[y][z];


    REPORT calculate_meta_coefficients ();

    REPORT calculate_stress_tau ();



    REPORT_USE( 
            prmt::Report report;
            report.result = _report .result;
            _return (report););
};

template<uint8_t dim>
void ElasticProblem2DOnCellV2<dim>::
print_result (const std::string &file_name)
{
    output_file_name = file_name;
    output_results ();
//    problem_of_torsion_rod .print_result (file_name);
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    puts("111");
    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    puts("112");
    std::ofstream output1 ("csp.1");
    c_sparsity .print_gnuplot (output1);

    puts("113");
    DomainLooper<dim, true> dl;
    REPORT dl .loop_domain(
            this->domain.dof_handler,
            black_on_white_substituter,
            c_sparsity);
//    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    puts("114");
    std::ofstream output2 ("csp.2");
    c_sparsity .print_gnuplot (output2);

    c_sparsity.compress ();

    puts("115");
    this->system_equations .reinit (c_sparsity, 
            this->domain.dof_handler.n_dofs());
//    printf("??????????\n");
    
//    for (size_t i = 0; i < dim + 1; ++i)
//        for (size_t j = i; j < dim + 1; ++j)
//        {
//            solution[i][j] .reinit (
//                    this->domain.dof_handler.n_dofs() +
//                    this->domain.dof_handler.n_dofs() / 2);
//            stress[i][j]   .reinit (
//                    this->domain.dof_handler.n_dofs() +
//                    this->domain.dof_handler.n_dofs() / 2);
//
//            solution[i][j] = 0.0;
//            stress[i][j] = 0.0;
//        };
//    printf("AAAAAAAAAAAAAAAAA\n");
    for (auto theta : {x, y, z})
        for (auto lambda : {x, y, z})
            if ((theta == lambda) or ((theta == x) and (lambda == y)))
            {
                solution[theta][lambda] .reinit (
                        this->domain.dof_handler.n_dofs());
                stress[theta][lambda]   .reinit (
                        this->domain.dof_handler.n_dofs()); 
            };

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::assemble_matrix_of_system ()
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
//        printf("AAAAAAAAAAAAAAA %f\n", this->system_equations.A.el(0,0));
    };
//    for (size_t i = 0; i < this->system_equations.A.m(); ++i)
//        for (size_t j = 0; j < this->system_equations.A.m(); ++j)
//            if (this->system_equations.A.el(i,j))
//                printf("A[%ld][%ld]=%f\n", i,j,this->system_equations.A.el(i,j));
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::
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
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::
assemble_right_vector_of_system_parallel (
                typename dealii::Vector<double>& b)
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

//                puts("11111111111d");
//                printf("%f\n", b(0));
        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
            b (local_dof_indices[i]) +=
                cell_rhs (i);
        };
    };
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::calculate_mean_coefficients ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    for (size_t i = 0; i < dim + 1; ++i)
        for (size_t j = 0; j < dim + 1; ++j)
            for (size_t k = 0; k < dim + 1; ++k)
                for (size_t l = 0; l < dim + 1; ++l)
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

    for (size_t i = 0; i < dim + 1; ++i)
        for (size_t j = 0; j < dim + 1; ++j)
            for (size_t k = 0; k < dim + 1; ++k)
                for (size_t l = 0; l < dim + 1; ++l)
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
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::calculate_stress_tau ()
{
    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients | 
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const uint8_t dofs_per_cell = finite_element.dofs_per_cell;
    const uint8_t dofs_per_cell_z = 4;//problem_of_torsion_rod.finite_element.dofs_per_cell;
    const uint8_t num_quad_points = quadrature_formula.size();
    const uint8_t num_quad_points_z = 4;

    typename dealii::DoFHandler<dim>::active_cell_iterator cellz =
        problem_of_torsion_rod.domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices_z (dofs_per_cell_z);

    printf("%d %d %d %d\n", this->domain.triangulation.n_active_cells(),
            problem_of_torsion_rod.domain.triangulation.n_active_cells(),
            dofs_per_cell, dofs_per_cell_z);

    size_t cell_num = 0;
    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);

        std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3> tau; 

        // FOR_I(0, dim + 1) FOR_J(0, dim + 1) FOR_K(0, dim + 1) FOR_L(0, dim + 1)
        for (auto i : {x, y, z}) 
            for (auto j : {x, y, z}) 
                for (auto k : {x, y, z}) 
                    for (auto l : {x, y, z}) 
            tau[i][j][k][l] = 0.0;

        cell->get_dof_indices (local_dof_indices);
        // cellz->get_dof_indices (local_dof_indices_z);

        // printf("%f\n", tau[0][0]);

        // puts("11111111111111111");
        for (auto i : {x, y, z}) 
            for (auto j : {x, y, z}) 
                for (auto k : {x, y, z}) 
                    for (auto l : {x, y, z}) 
                    {
                        // printf("TAU %f\n", tau[i][j][k][l]);
                        if (((i == j) and (k == l)) or 
                                ((i == x) and (j == y) and (k == x) and (l == y)))
                        {
                            // printf("%d %d %d %d\n", i, j, k, l);
                            FOR_N(0, dofs_per_cell)
                            {
                                for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
                                {
                                    FOR_O(0, dim)
                                        FOR_P(0, dim)
                                        {
                                            // if (count == 14)
                                            //     printf("%d %d %ld %ld %d %f\n", k, l, n, p,
                                            //             cell->vertex_dof_index(n, o),
                                            //     solution[k][l](cell->vertex_dof_index(n, 0)));
                                            tau[i][j][k][l] += 
                                                fe_values.shape_grad (n, q_point)[p] * 
                                                solution[k][l](local_dof_indices[n]) * //(cell->vertex_dof_index(n, o)) *
                                                coefficient[i][j][o][p][cell->material_id()] *
                                                fe_values.JxW(q_point);
                                        };
                                };
                            };
                            double area = 0.0;
                            FOR_O(0, num_quad_points)
                                area += fe_values.JxW(o);
                            tau[i][j][k][l] /= area; 
                            tau[i][j][k][l] += 
                                coefficient[i][j][k][l][cell->material_id()];
                        };
                        if (
                                ((i == z) and ((j == x) or (j == y))) and
                                ((k == z) and ((l == x) or (l == y)))
                           )
                        {
                            // printf("ijkl  %d %d %d %d\n", i, j, k, l);
                            tau[i][j][k][l] = problem_of_torsion_rod.flow[cell_num].second[j][l];
                        };
                           // FOR_N(0, dofs_per_cell_z)
                           // {
                           //     for (size_t q_point = 0; q_point < num_quad_points_z; ++q_point)
                           //     {
                           //         FOR_P(0, dim)
                           //         {
                           //             tau[i][j][k][l] += 
                           //                 fe_values.shape_grad (n, q_point)[p] * 
                           //                 problem_of_torsion_rod.solution[l](
                           //                         local_dof_indices_z[n]) *
                           //                 // coefficient[i][j][2][p][cell->material_id()] *
                           //                 fe_values.JxW(q_point);
                           //         };
                           //     };
                           // };

                    };
        // printf("%ld\n", count);
//        ++cellz;
        ++cell_num;

        // printf("%f\n", tau[0][0]);

        dealii::Point<2> midle(
                (cell->vertex(0)[0] +
                 cell->vertex(1)[0] +
                 cell->vertex(2)[0] +
                 cell->vertex(3)[0]) / 4.0,
                (cell->vertex(0)[1] +
                 cell->vertex(1)[1] +
                 cell->vertex(2)[1] +
                 cell->vertex(3)[1]) / 4.0);

        stress_tau.push_back(std::make_pair(midle, tau));
    };
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::calculate_meta_coefficients ()
{
    size_t len_vector_solution = this->domain.dof_handler.n_dofs();
    double mean_stress[dim + 1][dim + 1][dim + 1][dim + 1];

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
//    std::vector<
//        std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3> > 
//        tau (len_vector_solution / 2); 
    std::array<std::array<std::array<std::array<
        dealii::Vector<double>, 3>, 3>, 3>, 3> 
        tau; 

    for (auto i : {x, y, z}) 
        for (auto j : {x, y, z}) 
            for (auto k : {x, y, z}) 
                for (auto l : {x, y, z}) 
                {
                    double temp = 0.0;

                    for (size_t o = 0; o < 
                            len_vector_solution
                            + len_vector_solution / 2
                            ; ++o)
                    {
                        temp += 
                            human<Solution>(i, j, o) *
                            (-human<Stress>(k, l, o));
                    };
//                    for (size_t o = 0; o < len_vector_solution; ++o)
//                        temp += 

//                    if ((i == y) &&
//                        (j == y) &&
//                        (k == y) &&
//                        (l == y))
//                        printf("!!!!!!! %f %f %f\n", 
//                                (temp / area_of_domain), 
//                                mean_coefficient[i][j][k][l],
//                                (temp / area_of_domain) +
//                                mean_coefficient[i][j][k][l]);

                    meta_coefficient[i][j][k][l] =
                        mean_coefficient[i][j][k][l] + 
                        (temp / area_of_domain);

//                    tau[i][j][k][l] .reinit (len_vector_solution / 2);
//
//                    for (size_t o = 0; o < 
//                            len_vector_solution / 2
//                            ; ++o)
//                    {
//                        tau[i][j][k][l](o) = 
//                            human<Solution>(i, j, 2 * o) *
//                            (-human<Stress>(k, l, 2 * o)) +
//                            human<Solution>(i, j, 2 * o + 1) *
//                            (-human<Stress>(k, l, 2 * o + 1)) +
//                            human<Solution>(i, j, len_vector_solution + o) *
//                            (-human<Stress>(k, l, len_vector_solution + o));
//                    };
                };



//    {
//        char suffix[3] = {'x', 'y', 'z'};
//
//        for (auto i : {x, y, z}) 
//            for (auto j : {x, y, z}) 
//                for (auto k : {x, y, z}) 
//                    for (auto l : {x, y, z}) 
//        {
//            dealii::DataOut<dim> data_out;
//            data_out.attach_dof_handler (problem_of_torsion_rod.domain.dof_handler);
//
//            data_out.add_data_vector (tau[i][j][k][l], "solution");
//            data_out.build_patches ();
//
//            std::string file_name = "./tau/";
//            file_name += suffix[i];
//            file_name += suffix[j];
//            file_name += suffix[k];
//            file_name += suffix[l];
//            file_name += ".gpd";
//
//            std::ofstream output (file_name.data());
//            data_out.write_gnuplot (output);
//        };
//    };

//    for (auto i : {x, y, z}) 
//        for (auto j : {x, y, z}) 
//        {
//    meta_coefficient[i][j][z][x] = meta_coefficient[x][z][x][z];
//        };
//
//    for (size_t i = 0; i < dim + 1; ++i)
//        for (size_t j = 0; j < dim + 1; ++j)
//            for (size_t k = 0; k < dim + 1; ++k)
//                for (size_t l = 0; l < dim + 1; ++l)
//                    meta_coefficient[i][j][k][l] = 
//                        mean_coefficient[i][j][k][l];
//
////    reducing to 2d view
//
//    uint8_t width_2d_matrix = dim * dim;
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / dim;
//        uint8_t in = i % dim;
//
//        for (size_t j = i; j < width_2d_matrix; ++j)
//        {
//            uint8_t jm = j / dim;
//            uint8_t jn = j % dim;
//
////            printf("||||||||| %d %d %d %d\n", im, in, jm, jn);
//
//            mean_stress[im][in][jm][jn] = 0.0;
//
//            uint8_t a = im;
//            uint8_t b = in;
//            uint8_t c = jm;
//            uint8_t d = jn;
//
//            if (im == 1)
//                a ^= b ^= a ^= b;
//
//            if (jm == 1)
//                c ^= d ^= c ^= d;
//
//            for (size_t k = 0; k < len_vector_solution; ++k)
//            {
//                mean_stress[im][in][jm][jn] += 
//                    solution[a][b](k) * (-stress[c][d](k));
//            if ((i == 0) and (j==0))
//                printf("%f %f\n", solution[a][b](k) , stress[c][d](k));
//            };

//            if ((i == 0) and (j==0))
//            printf("STRESS=%f\n", mean_stress[im][in][jm][jn]);

//            mean_stress[im][in][jm][jn] /= area_of_domain; 
//
//            meta_coefficient[im][in][jm][jn] += 
//                mean_stress[im][in][jm][jn];
//
//            meta_coefficient[jm][jn][im][in] = 
//                meta_coefficient[im][in][jm][jn];
//        };
//    };
//
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / dim;
//        uint8_t in = i % dim;
//
//        mean_stress[im][in][z][z] = 0.0;
//
//        uint8_t a = im;
//        uint8_t b = in;
//
//        if (im == 1)
//            a ^= b ^= a ^= b;
//
//        for (size_t k = 0; k < len_vector_solution; ++k)
//            mean_stress[im][in][z][z] += 
//                solution[a][b](k) * (-this->system_equations.b(k));
//
//        mean_stress[im][in][z][z] /= area_of_domain; 
//
//        meta_coefficient[im][in][z][z] += 
//            mean_stress[im][in][z][z];
//
//        meta_coefficient[z][z][im][in] = 
//            meta_coefficient[im][in][z][z];
//    };
//
//    const uint8_t xx = 0;
//    const uint8_t yy = 1;
//    const uint8_t xy = 2;
//
//    meta_coefficient[x][z][x][z] = problem_of_torsion_rod .meta_coefficient[xx];
//    meta_coefficient[y][z][y][z] = problem_of_torsion_rod .meta_coefficient[yy];
//    meta_coefficient[x][z][y][z] = problem_of_torsion_rod .meta_coefficient[xy];
//
//    meta_coefficient[z][x][z][x] = meta_coefficient[x][z][x][z];
//    meta_coefficient[z][y][z][y] = meta_coefficient[y][z][y][z];
//    meta_coefficient[z][x][z][y] = meta_coefficient[x][z][y][z];
//    meta_coefficient[z][y][z][x] = meta_coefficient[x][z][y][z];
//    meta_coefficient[y][z][x][z] = meta_coefficient[x][z][y][z];
//    meta_coefficient[z][x][x][z] = meta_coefficient[x][z][x][z];
//    meta_coefficient[z][y][y][z] = meta_coefficient[y][z][y][z];
//    meta_coefficient[z][x][y][z] = meta_coefficient[x][z][y][z];
//    meta_coefficient[z][y][x][z] = meta_coefficient[y][z][x][z];
//    meta_coefficient[x][z][z][x] = meta_coefficient[x][z][x][z];
//    meta_coefficient[y][z][z][y] = meta_coefficient[y][z][y][z];
//    meta_coefficient[x][z][z][y] = meta_coefficient[x][z][y][z];
//    meta_coefficient[y][z][z][x] = meta_coefficient[y][z][x][z];
//
//    meta_coefficient[z][z][z][z] = meta_coefficient[y][y][y][y];


    /////////////////////////////

//    for (size_t k = 0; k < coefficient[z][z][z][z].size(); ++k)
//        meta_coefficient[z][z][z][z] +=
//            coefficient[z][z][z][z][k] * 
//            area_of_material[k];
//
//    meta_coefficient[z][z][z][z] /= area_of_domain;

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
   
//    puts("xx, xy, xz, yx, yy, yz, zx, zy, zz");
//    FOR_I(0, this->domain.dof_handler.n_dofs() + 
//            this->domain.dof_handler.n_dofs() / 2)
//    {
//        if (i == this->domain.dof_handler.n_dofs())
//            printf("\x1B[31m%f %f %f %f %f %f %f %f %f\x1B[0m\n", 
//                    human<Stress>(x, x, i),
//                    human<Stress>(x, y, i),
//                    human<Stress>(x, z, i),
//                    human<Stress>(y, x, i),
//                    human<Stress>(y, y, i),
//                    human<Stress>(y, z, i),
//                    human<Stress>(z, x, i),
//                    human<Stress>(z, y, i),
//                    human<Stress>(z, z, i));
//        else
//            printf("%f %f %f %f %f %f %f %f %f\n", 
//                    human<Stress>(x, x, i),
//                    human<Stress>(x, y, i),
//                    human<Stress>(x, z, i),
//                    human<Stress>(y, x, i),
//                    human<Stress>(y, y, i),
//                    human<Stress>(y, z, i),
//                    human<Stress>(z, x, i),
//                    human<Stress>(z, y, i),
//                    human<Stress>(z, z, i));
//    };
//
//    puts(" ");
//    puts("xx, xy, xz, yx, yy, yz, zx, zy, zz");
//    FOR_I(0, this->domain.dof_handler.n_dofs() + 
//            this->domain.dof_handler.n_dofs() / 2)
//    {
//        if (i == this->domain.dof_handler.n_dofs())
//            printf("\x1B[31m%f %f %f %f %f %f %f %f %f\x1B[0m\n", 
//                    human<Solution>(x, x, i),
//                    human<Solution>(x, y, i),
//                    human<Solution>(x, z, i),
//                    human<Solution>(y, x, i),
//                    human<Solution>(y, y, i),
//                    human<Solution>(y, z, i),
//                    human<Solution>(z, x, i),
//                    human<Solution>(z, y, i),
//                    human<Solution>(z, z, i));
//        else
//            printf("%f %f %f %f %f %f %f %f %f\n", 
//                    human<Solution>(x, x, i),
//                    human<Solution>(x, y, i),
//                    human<Solution>(x, z, i),
//                    human<Solution>(y, x, i),
//                    human<Solution>(y, y, i),
//                    human<Solution>(y, z, i),
//                    human<Solution>(z, x, i),
//                    human<Solution>(z, y, i),
//                    human<Solution>(z, z, i));
//    };
//
//    printf("%f %f %f\n", area_of_material[0], area_of_material[1], area_of_domain);

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::solve_system_equations ()
{
    for (size_t i = 0; i < this->system_equations.A.m(); ++i)
        for (size_t j = 0; j < this->system_equations.A.n(); ++j)
            if (i != j)
                if (
                        (this->system_equations.A .el (i,j)) and
                        (this->system_equations.A .el (j,i)))
                    if (fabs (
                                this->system_equations.A .el (i,j) - 
                                this->system_equations.A .el (j,i)
                             ) > 1e-10)
                        printf("\x1B[31mWARNING %ld %ld\x1B[0m\n", i, j);

    dealii::SolverControl solver_control (1000000, 1e-8);

//    Femenist::SolverSE<> solver(solver_control);

    dealii::SolverCG<> solver(solver_control);

    this->system_equations.x = 0;

//    REPORT solver .solve (
//            this->system_equations.A,
//            this->system_equations.x,
//            this->system_equations.b);

    solver .solve (
            this->system_equations.A,
            this->system_equations.x,
            this->system_equations.b
            ,dealii::PreconditionIdentity());
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>::solve_system_equations_parallel (
                typename dealii::Vector<double>& x, 
                typename dealii::Vector<double>& b)
{
    for (size_t i = 0; i < this->system_equations.A.m(); ++i)
        for (size_t j = 0; j < this->system_equations.A.n(); ++j)
            if (i != j)
                if (
                        (this->system_equations.A .el (i,j)) and
                        (this->system_equations.A .el (j,i)))
                    if (fabs (
                                this->system_equations.A .el (i,j) - 
                                this->system_equations.A .el (j,i)
                             ) > 1e-10)
                        printf("\x1B[31mWARNING %ld %ld\x1B[0m\n", i, j);

    // FILE* F;
    // F = fopen("A1.gpd", "w");
    // for (size_t i = 0; i < this->system_equations.A.m(); ++i)
    //     for (size_t j = 0; j < this->system_equations.A.n(); ++j)
    //         if (std::abs(this->system_equations.A .el (i, j)) > 1e-6)
    //             fprintf(F, "%ld %ld %f\n", i, j, this->system_equations.A .el (i, j));
    //         else
    //             fprintf(F, "%ld %ld 0.0\n", i, j);
    // fclose(F);

    dealii::SolverControl solver_control (100000, 1e-8);

    dealii::SolverCG<> solver(solver_control);
//    Femenist::SolverSE<> solver(solver_control);

    this->system_equations.x = 0;

    solver .solve (
            this->system_equations.A,
            x,
            b
            ,dealii::PreconditionIdentity());
//    );
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report ElasticProblem2DOnCellV2<dim>:: output_results ()
{
    for (auto theta : {x, y, z})
        for (auto lambda : {x, y, z})
            if ((theta == lambda) or ((theta == x) and (lambda == y)))
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

            data_out.add_data_vector (solution[theta][lambda], solution_names);
            data_out.build_patches ();

            std::string file_name = output_file_name;
            file_name += suffix[theta];
            file_name += suffix[lambda];
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };

    problem_of_torsion_rod .print_result (output_file_name + "z");

    {
        // for (auto i : {x, y, z}) 
        //     for (auto j : {x, y, z}) 
        //         for (auto k : {x, y, z}) 
        //             for (auto l : {x, y, z}) 
        //                 FOR_N(0, stress_tau.size())
        //                     printf("2 TAU %f\n", stress_tau[n].second[i][j][k][l]);

        double meta[3][3][3][3] = {0.0};
        char suffix[3] = {'x', 'y', 'z'};
        FOR_I(0, dim + 1)
            FOR_J(0, dim + 1)
            {
                std::string file_name = "stress_";
                file_name += suffix[i];
                file_name += suffix[j];
                file_name += ".gpd";
                FILE *F;
                F = fopen(file_name.data(), "w");
                FOR_N(0, stress_tau.size())
                {
                    fprintf(F, "%f %f  %f %f %f  %f %f %f  %f %f %f\n", 
                            stress_tau[n].first(0),
                            stress_tau[n].first(1),
                            stress_tau[n].second[i][j][0][0],
                            stress_tau[n].second[i][j][0][1],
                            stress_tau[n].second[i][j][0][2],
                            stress_tau[n].second[i][j][1][0],
                            stress_tau[n].second[i][j][1][1],
                            stress_tau[n].second[i][j][1][2],
                            stress_tau[n].second[i][j][2][0],
                            stress_tau[n].second[i][j][2][1],
                            stress_tau[n].second[i][j][2][2]);
                    FOR_K(0, dim + 1)
                        FOR_L(0, dim + 1)
                        meta[i][j][k][l] += stress_tau[n].second[i][j][k][l];
                };
                fclose(F);
            };
        FOR_I(0, dim + 1)
            FOR_J(0, dim + 1)
                    FOR_K(0, dim + 1)
                        FOR_L(0, dim + 1)
                        printf("META[%ld][%ld][%ld][%ld] = %f \n", i, j, k, l, meta[i][j][k][l] / stress_tau.size());

    };

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
typename HeatConductionProblemSup<dim>::TypeCoef
ElasticProblem2DOnCellV2<dim>::coef_for_problem_of_torsion_rod (
        const typename ElasticProblemSup<dim+1>::TypeCoef &coef) const
{
    typename HeatConductionProblemSup<dim>::TypeCoef coef_temp;

    const uint8_t XX = 0;
    const uint8_t YY = 1;
    const uint8_t XY = 2;

    coef_temp[xx] .clear ();
    coef_temp[yy] .clear ();
    coef_temp[xy] .clear ();

    for (size_t i = 0; i < coef[0][0][0][0].size(); ++i)
    {
        coef_temp[XX] .push_back (coef[x][z][x][z][i]);
        coef_temp[YY] .push_back (coef[y][z][y][z][i]);
        coef_temp[XY] .push_back (coef[x][z][y][z][i]);
    };

    return coef_temp;
};

template<uint8_t dim>
template<uint8_t type_res>
double ElasticProblem2DOnCellV2<dim>::human (
        uint8_t theta, uint8_t lambda, size_t index) const
{
    double res;

    uint8_t comp = theta * 3 + lambda;

    if (type_res == Solution)
    {
        switch (comp)
        {
            case xx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return solution[x][x](index);
                    else
                        return 0.0;
                };
            case xy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return solution[x][y](index);
                    else
                        return 0.0;
                };
            case xz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.solution[x](n);
                    };
                };
            case yx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return solution[x][y](index);
                    else
                        return 0.0;
                };
            case yy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return solution[y][y](index);
                    else
                        return 0.0;
                };
            case yz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.solution[y](n);
                    };
                };
            case zx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.solution[x](n);
                    };
                };
            case zy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.solution[y](n);
                    };
                };
            case zz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return solution[z][z](index);
                    else
                        return 0.0;
                };
        };
    }
    else if (type_res == Stress)
    {
        switch (comp)
        {
            case xx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return stress[x][x](index);
                    else
                        return 0.0;
                };
            case xy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return stress[x][y](index);
                    else
                        return 0.0;
                };
            case xz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.heat_flow[x](n);
                    };
                };
            case yx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return stress[x][y](index);
                    else
                        return 0.0;
                };
            case yy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return stress[y][y](index);
                    else
                        return 0.0;
                };
            case yz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.heat_flow[y](n);
                    };
                };
            case zx:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.heat_flow[x](n);
                    };
                };
            case zy:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return 0.0;
                    else
                    {
                        size_t n = index - this->domain.dof_handler.n_dofs();
                        return problem_of_torsion_rod.heat_flow[y](n);
                    };
                };
            case zz:
                {
                    if (index < this->domain.dof_handler.n_dofs())
                        return stress[z][z](index);
                    else
                        return 0.0;
                };
        };
    };

    return res;
};

#endif









