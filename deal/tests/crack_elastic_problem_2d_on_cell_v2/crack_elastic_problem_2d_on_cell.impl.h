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

//#include "./crack_elastic_problem_2d_on_cell.desc.h"

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
        const typename ElasticProblemSup<dim + 1>::TypeCoef &coef,
        const vec<prmt::LoopCondition<dim>> &loop_border)
:
    Problem< 
        dim,
        ElementStiffnessMatrixElasticProblem<dim>,
        ElementRightHandSideVectorElasicProblemOnCell<dim> > (),
    finite_element (dealii::FE_Q<dim>(1), dim),
    problem_of_torsion_rod (triangulation, coef_for_problem_of_torsion_rod (coef)),
    complete_main_stress_1(triangulation.n_active_cells()),
    complete_main_stress_2(triangulation.n_active_cells()),
    complete_stress(triangulation.n_active_cells()),
    loop_condition (loop_border)
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
Report ElasticProblem2DOnCellV2<dim>::solved ()
{
    REPORT setup_system ();
    REPORT assemble_matrix_of_system ();
    REPORT calculate_mean_coefficients ();

    // 2d elastic problem 
    
    std::array<
        std::array<
        std::vector<double>,
        dim>,
        dim> coef_for_rhs;

    for (auto i : {0, 1})//range<0, 1>())
        for (auto j : {0, 1})//range<0, 1>())
            coef_for_rhs[i][j] .resize (coefficient[i][j][0][0].size());

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

                this->element_rh_vector .set_coefficient (coef_for_rhs);

                REPORT assemble_right_vector_of_system_parallel (
                        assigned_to stress[theta][lambda]);
            };

    REPORT problem_of_torsion_rod .setup_system ();
    REPORT problem_of_torsion_rod .assemble_matrix_of_system ();
    REPORT problem_of_torsion_rod .calculate_mean_coefficients ();
    REPORT problem_of_torsion_rod .calculate_cells_area ();
    
    {
        std::array<std::vector<double>, dim> coef_for_rhs;
        for(size_t i = 0; i < dim; ++i)
            coef_for_rhs[i] .resize (problem_of_torsion_rod .coefficient[i].size());

        for(size_t i = 0; i < dim; ++i)
        {
            for(size_t j = 0; j < dim; ++j)
            {
                for(size_t k = 0; k < problem_of_torsion_rod .coefficient[problem_of_torsion_rod .conver(i,j)].size(); ++k)
                    coef_for_rhs[j][k] = problem_of_torsion_rod .coefficient[problem_of_torsion_rod .conver(i,j)][k];
            };

            problem_of_torsion_rod .solution[i]  = 0;
            problem_of_torsion_rod .heat_flow[i] = 0;

            problem_of_torsion_rod .element_rh_vector .set_coefficient (coef_for_rhs);

            REPORT problem_of_torsion_rod .assemble_right_vector_of_system_parallel (assigned_to problem_of_torsion_rod .heat_flow[i]);
        };
    };

    bool break_flag = false;
    max_complete_main_stress_1 .push_back(1e+10);
    {
        arr<dbl,3> temp = {0.0};
        fiz_coef .push_back(temp);
    };
    vec<st> cll(1);
    // cll[0] = 550;
    // miscarry_broken_cells (cll);
    REPORT calculate_cells_area ();
    if (0)
    {
        {
            FILE *F;
            F = fopen("brocken_cells_in_line.gpd", "r");
            st num_rows = 0;
            fscanf(F,"%ld", &num_rows);
            FOR_I(0, num_rows)
            {
                dbl x = 0.0;
                dbl y = 0.0;
                dbl nul = 0.0;
                dbl array = 0.0;
                st id_cell = 0;
                fscanf(F, "%lf %lf %lf %lf %ld", &x, &y, &nul, &array, &id_cell);
                brocken_cell_in_line .push_back (id_cell);
                brocken_cell_in_line_coor .push_back (dealii::Point<dim>(x,y));
                printf("%ld %ld %f\n", num_rows, id_cell, x);
            };
            miscarry_broken_cells (brocken_cell_in_line);
            fclose(F);
        };

        {
            FILE *F;
            F = fopen("max_stress.gpd", "r");
            st num_rows = 0;
            fscanf(F,"%ld", &num_rows);
            FOR_I(0, num_rows)
            {
                dbl max = 0.0;
                arr<dbl, 3> fiz = {0.0};
                fscanf(F, "%lf %lf %lf %lf", &max, &fiz[0], &fiz[1], &fiz[2]);
                max_complete_main_stress_1 .push_back (max);
                fiz_coef.push_back (fiz);
                printf("%ld %F %f %f %f\n", num_rows, max, fiz[0], fiz[1], fiz[2]);
            };
            fclose(F);
        };
    };

    FOR_I(0, 1)
    {
        vec<vec<st>> cells_i; 
        // FOR_J(1, 2) //78
        FOR_J(0, 1) //78
        {
            printf("LOOP %ld %ld\n", i, j);
#pragma omp parallel for
            for (size_t theta = 0; theta < 3; ++theta)
                for (size_t lambda = 0; lambda < 3; ++lambda)
                    if ((theta == lambda) or ((theta == x) and (lambda == y)))
                    {
                        REPORT solve_system_equations_parallel (
                                solution[theta][lambda], stress[theta][lambda]);

                        FOR_O(0, this->system_equations.x.size()) 
                            solution[theta][lambda](o) = solution[theta][lambda](
                                    black_on_white_substituter .subst (o));
                    };

#pragma omp parallel for
            for(size_t i = 0; i < dim; ++i)
            {
                REPORT problem_of_torsion_rod .solve_system_equations_parallel (
                        problem_of_torsion_rod .solution[i], problem_of_torsion_rod .heat_flow[i]);

                for (size_t j = 0; j < problem_of_torsion_rod .system_equations.x.size(); ++j)
                    problem_of_torsion_rod .solution[i](j) = problem_of_torsion_rod .solution[i](
                            problem_of_torsion_rod .black_on_white_substituter .subst (j));
            };
            REPORT problem_of_torsion_rod .calculate_meta_coefficients ();
            REPORT problem_of_torsion_rod .calculate_flow ();
            // problem_of_torsion_rod .solved ();

            // REPORT calculate_mean_coefficients ();

            REPORT calculate_cells_stress ();

            REPORT calculate_meta_coefficients ();

            REPORT calculate_main_stress ();

            REPORT print_stress (i, j);

            vec<st> cells_j;
            dbl maximum_stress = compute_max_complete_main_stress_1  (cells_j);
            printf("cell_j size %d\n", cells_j.size());
            printf("max %f\n", maximum_stress);
            cells_i .push_back (cells_j);

            miscarry_broken_cells (cells_j);

            // printf("maximu_stress %f\n", maximum_stress);
            // if (maximum_stress < max_complete_main_stress_1 .back())
            // {
                max_complete_main_stress_1  .push_back (maximum_stress);
            //     break;
            // };

            // if (at_boundary(cells_j))
            // {
            //     break_flag = true;
            // };

        };

        brocken_cell .push_back (cells_i);

        if (break_flag == true)
            break;
    };



    REPORT_USE( 
            Report report;
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

template<u8 dim>
arr<dbl, 3> ElasticProblem2DOnCellV2<dim>::calculate_phisical_meta ()
{
    arr<dbl, 3> res;

    enum {x, y, z};

    dbl unphys[dim + 1][dim + 1][dim + 1][dim + 1];
    FOR_I(0, 3) FOR_J(0, 3) FOR_K(0, 3) FOR_L(0, 3)
        unphys[i][j][k][l] = meta_coefficient[i][j][k][l];

    double A = 
        unphys[x][x][x][x] * unphys[y][y][y][y] * unphys[z][z][z][z] - 
        unphys[y][y][z][z] * unphys[z][z][y][y] * unphys[x][x][x][x] +
        unphys[x][x][y][y] * unphys[y][y][z][z] * unphys[z][z][x][x] - 
        unphys[y][y][x][x] * unphys[x][x][y][y] * unphys[z][z][z][z] - 
        unphys[y][y][y][y] * unphys[x][x][z][z] * unphys[z][z][x][x] +
        unphys[y][y][x][x] * unphys[x][x][z][z] * unphys[z][z][y][y]; 

    double B =
        unphys[y][y][y][y] * unphys[z][z][z][z] - 
        unphys[y][y][z][z] * unphys[z][z][y][y]; 

    dbl Exxxx = A / B;
    dbl Nyx   = 
        (unphys[y][y][x][x] * unphys[z][z][z][z] - 
        unphys[z][z][x][x] * unphys[y][y][z][z]) / B; 
    dbl Nzx   = -
        (unphys[y][y][x][x] * unphys[z][z][y][y] - 
        unphys[y][y][y][y] * unphys[z][z][x][x]) / B; 

    printf("%f %f %f\n", Exxxx, Nyx, Nzx);
    arr<dbl, 3> fiz = {Exxxx, Nyx, Nzx};
    fiz_coef.push_back(fiz);

    res[0] = 1 / Exxxx;
    res[1] = - Nyx / Exxxx;
    res[2] = - Nzx / Exxxx;

//    printf("%f %f %f\n", Exxxx, Nyx, Nzx);
    return res;
};

template<u8 dim>
Report ElasticProblem2DOnCellV2<dim>::miscarry_broken_cells (vec<st> &cells)
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

    dealii::FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    FOR_N (0, cells.size())
    {
        typename dealii::DoFHandler<dim>::active_cell_iterator cell =
            this->domain.dof_handler.begin_active();
        FOR_I(0, cells[n])
            ++cell;

        fe_values .reinit (cell);
        cell_matrix = 0;

        for (size_t i = 0; i < dofs_per_cell; ++i)
            for (size_t j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i,j) = this->element_stiffness_matrix
                    (i, j, quadrature_formula, fe_values, cell->material_id()); 

        cell ->get_dof_indices (local_dof_indices);


        // for (size_t i = 0; i < dofs_per_cell; ++i)
        //     local_dof_indices[i] = black_on_white_substituter .subst (
        //             local_dof_indices[i]);


        for (size_t i = 0; i < dofs_per_cell; ++i)
            for (size_t j = 0; j < dofs_per_cell; ++j)
               this->system_equations.A .add (local_dof_indices[i],
                                              local_dof_indices[j],
                                              -cell_matrix(i,j));
        // break;
    };
    REPORT_USE( 
            Report report;
            report.result = _report .result;
            _return (report););
};

template<u8 dim>
Report ElasticProblem2DOnCellV2<dim>::compute_borders ()
{
    const uint8_t y = 1;
    const uint8_t z = 2;
    const uint8_t wht = 0;
    const uint8_t blc = 1;


    for (uint8_t i = 0; i < dim; ++i)
    {
        border.coor[i][wht] =  0xffffffff * 1.0;
        border.coor[i][blc] = -0xffffffff * 1.0;
    };

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        if (cell->at_boundary())
            for(uint8_t i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                for (uint8_t j = 0; j < dim; ++j)
                {
                    if (cell->vertex(i)[j] < border.coor[j][wht])
                        border.coor[j][wht] = cell->vertex(i)[j];
                    if (cell->vertex(i)[j] > border.coor[j][blc])
                        border.coor[j][blc] = cell->vertex(i)[j];
                };
    };

    REPORT_USE( 
            Report report;
            report.result = _report .result;
            _return (report););
};


template<u8 dim>
bool ElasticProblem2DOnCellV2<dim>::at_boundary (vec<st> &cells)
{
    dbl MIN_DISTANCE = 1e-10;

    FOR_N (0, cells.size())
    {
        typename dealii::DoFHandler<dim>::active_cell_iterator cell =
            this->domain.dof_handler.begin_active();
        FOR_I(0, cells[n])
            ++cell;

        FOR_I (0, 4)
        {
            FOR_J (0, 2)
            {
                if (
                        (fabs(cell->vertex(i)[j] - border.coor[j][0]) < MIN_DISTANCE) or
                        (fabs(cell->vertex(i)[j] - border.coor[j][1]) < MIN_DISTANCE)
                   )
                    return true;
            };
        };
    };
};

template<u8 dim>
dbl ElasticProblem2DOnCellV2<dim>::compute_max_complete_main_stress_1 (vec<st> &cells)
{
    dbl max = 0.0;


    // FILE* f;
    // f = fopen("str_yy.gpd", "w");

    // dbl integ = 0.0;
    dbl sum = 0.0;
//     for (dbl stress : complete_main_stress_1)
//     {
// //         dbl stress = -
// //             (E[0] * cells_stress[i].second[y][y][x][x] +
// //              E[1] * cells_stress[i].second[y][y][y][y] +
// //              E[2] * cells_stress[i].second[y][y][z][z]);
// // //        printf("%f\n", stress);
// //         fprintf(f, "%f %f %f\n", 
// //                 cells_stress[i].first(0),
// //                 cells_stress[i].first(1),
// //                 stress);
//         if (max < stress)
//             max = stress;
//         // integ += stress;
//         sum += stress; //_tau[i].second[x][x][y][y]; 
//     };
    // fprintf(f, "%f %f %f %f %f\n", integ / cells_stress.size(), max, E[0], E[1], E[2]);
    // fclose(f);
    // printf("cell_j size %d\n", cells.size());
    FOR_I(0, complete_main_stress_1.size())
    {
        if ((max < complete_main_stress_1[i]) and (material_of_cell[i] == 0))
            max = complete_main_stress_1[i];
    };

    FOR_I(0, cells_stress.size())
    {
        // dbl stress = -
        //     (E[0] * cells_stress[i].second[y][y][x][x] +
        //      E[1] * cells_stress[i].second[y][y][y][y] +
        //      E[2] * cells_stress[i].second[y][y][z][z]);
        if (material_of_cell[i] == 0)
            if (fabs(max - complete_main_stress_1[i]) < 1e-2)
            {
                cells .push_back (i);
                brocken_cell_in_line .push_back (i);
                problem_of_torsion_rod .brocken_cell_in_line .push_back (i);
            };
    };
    // printf("max %f, sum %f, num_cell %d\n", max, sum / cells_stress.size(), cells.size());

    // f = fopen ("stupid_

    return max;
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    {
    st a = this->domain.dof_handler.locally_owned_dofs().n_elements();
    st b = this->domain.dof_handler.n_dofs();
    st c = this->domain.dof_handler.get_tria().get_vertices().size();
    st d = this->domain.dof_handler.get_tria().get_used_vertices().size();
    printf("abcd %ld %ld %ld %ld\n", a, b, c, d);
    FOR(i, 0, c)
        printf("%f %f %d %d\n",
                this->domain.dof_handler.get_tria().get_vertices()[i](0),
                this->domain.dof_handler.get_tria().get_vertices()[i](1),
                this->domain.dof_handler.locally_owned_dofs().nth_index_in_set(i*2),
                this->domain.dof_handler.locally_owned_dofs().nth_index_in_set(i*2+1));

    printf("\n");

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        FOR (n, 0, dealii::GeometryInfo<dim>::vertices_per_cell)
            printf("%f %f %d %d\n",
                    cell->vertex(n)(0),
                    cell->vertex(n)(1),
                    cell->vertex_dof_index (n, 0),
                    cell->vertex_dof_index (n, 1));
    };
};


    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    puts("111");
    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    puts("112");
    std::ofstream output1 ("csp.1");
    c_sparsity .print_gnuplot (output1);

    puts("113");
    prmt::DomainLooper<dim, 2> dl(loop_condition);
    // DomainLooper<dim, 2> dl;
    REPORT dl .loop_domain(
            this->domain.dof_handler,
            black_on_white_substituter,
            c_sparsity);
FILE *F;
F = fopen("bows_2","w");
FOR(i, 0, black_on_white_substituter.size)
    fprintf(F, "%ld %ld\n", 
            black_on_white_substituter.white[i],
            black_on_white_substituter.black[i]);
fclose(F);
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
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::assemble_matrix_of_system ()
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
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::
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
Report ElasticProblem2DOnCellV2<dim>::
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
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_mean_coefficients ()
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

    st cell_num = 0;
    for (; cell != endc; ++cell)
    {
        bool cell_exist = true;
        // for (auto i : brocken_cell_in_line)
        // {
        //     if (cell_num == i)
        //     {
        //         cell_exist = false;
        //     };
        // };
        ++cell_num;

        // if (cell_exist)
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
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_main_stress ()
{
    arr<dbl, 3> E = calculate_phisical_meta ();
    // { - 1.0 / meta_coefficient[x][x][x][x],
    //     meta_coefficient[x][x][y][y] / meta_coefficient[x][x][x][x],
    //     meta_coefficient[x][x][z][z] / meta_coefficient[x][x][x][x]}; 

    printf("E %f %f %f\n", E[0], E[1], E[2]);

    enum {x, y, z};
    
    FOR_I(0, cells_stress.size())
    {
        dbl str_xx = -
            (E[0] * cells_stress[i].second[x][x].problem[x][x] +
             E[1] * cells_stress[i].second[x][x].problem[y][y] +
             E[2] * cells_stress[i].second[x][x].problem[z][z]);

        dbl str_xy = -
            (E[0] * cells_stress[i].second[x][y].problem[x][x] +
             E[1] * cells_stress[i].second[x][y].problem[y][y] +
             E[2] * cells_stress[i].second[x][y].problem[z][z]);

        dbl str_yx = -
            (E[0] * cells_stress[i].second[y][x].problem[x][x] +
             E[1] * cells_stress[i].second[y][x].problem[y][y] +
             E[2] * cells_stress[i].second[y][x].problem[z][z]);

        dbl str_yy = -
            (E[0] * cells_stress[i].second[y][y].problem[x][x] +
             E[1] * cells_stress[i].second[y][y].problem[y][y] +
             E[2] * cells_stress[i].second[y][y].problem[z][z]);

        dbl L1 = str_xx + str_yy;
        dbl L2 = str_xx * str_yy - str_xy * str_yx;

        // dbl m_str_1 = (L1 - sqrt(L1 * L1 - 4.0 * L2)) / 2.0;
        // dbl m_str_2 = (L1 + sqrt(L1 * L1 - 4.0 * L2)) / 2.0;

        dbl m_str_1 = ((str_xx + str_yy) + sqrt(pow(str_xx - str_yy, 2.0) + 4.0 * str_xy * str_yx)) / 2.0;
        dbl m_str_2 = ((str_xx + str_yy) - sqrt(pow(str_xx - str_yy, 2.0) + 4.0 * str_xy * str_yx)) / 2.0;

        complete_main_stress_1[i] = m_str_1 > m_str_2 ? m_str_1 : m_str_2;
        complete_main_stress_2[i] = m_str_1 > m_str_2 ? m_str_2 : m_str_1;

        complete_stress[i][x][x] = str_xx;
        complete_stress[i][x][y] = str_xy;
        complete_stress[i][y][x] = str_yx;
        complete_stress[i][y][y] = str_yy;
    };
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report);); 
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_cells_area ()
{
    cells_area .clear ();

    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const uint8_t num_quad_points = quadrature_formula.size();

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);
        dbl area = 0.0;
        FOR_I(0, num_quad_points)
            area += fe_values.JxW(i);
        cells_area.push_back(area);

        material_of_cell.push_back(cell->material_id());
        // printf("AREA %f\n", area);
    };

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_cells_stress ()
{
    cells_stress .clear ();

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

    // printf("%d %d %d %d\n", this->domain.triangulation.n_active_cells(),
    //         problem_of_torsion_rod.domain.triangulation.n_active_cells(),
    //         dofs_per_cell, dofs_per_cell_z);

    size_t cell_num = 0;
    for (; cell != endc; ++cell)
    {
        arr<arr<ValuesProblems, num_cources>, num_cources> stress; 
        for (auto i : {x, y, z}) 
            for (auto j : {x, y, z}) 
                for (auto k : {x, y, z}) 
                    for (auto l : {x, y, z}) 
                        stress[i][j].problem[k][l] = 0.0;
        // printf("steress1 %f %f\n", 
        //         stress[y][z].problem[x][x],
        //         stress[x][x].problem[x][x]);

        // printf("%d\n", b_cell.size());
        bool cell_exist = true;
        for (auto i : brocken_cell_in_line)
        {
            // printf("%d\n", i);
            if (cell_num == i)
            {
                cell_exist = false;
            };
        };

        if ((cell_exist)) // and (cell->material_id() == 0))
        {
            fe_values .reinit (cell);

            dbl area = 0.0;
            FOR_O(0, num_quad_points)
                area += fe_values.JxW(o);

            dbl mat_id = cell->material_id();

            // cell->get_dof_indices (local_dof_indices);

            for (auto k : {x, y, z}) 
            for (auto l : {x, y, z}) 
            if ((k == l) or ((k == x) and (l == y)))
            {
                arr<arr<dbl,2>,2> deform;

                for (auto i : {x, y}) 
                    for (auto j : {x, y}) 
                    {
                        deform[i][j] = 0.0;

                        FOR_N(0, 4)
                        {
                            dbl temp = 0.0;
                            for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
                                temp += 
                                    fe_values.shape_grad ((n * 2 + i), q_point)[j] *
                                    fe_values.JxW(q_point);
                            deform[i][j] += 
                                solution[k][l](cell->vertex_dof_index(n, i)) * 
                                temp;
                                // (temp > -1e-10 ? 2.0 : -2.0);
                        };
                        deform[i][j] /= area;
                    };
                // if (k == z)
                //     printf("ZZZ %f %f %f %f\n", 
                //             deform[x][x],
                //             deform[x][y],
                //             deform[y][x],
                //             deform[y][y]
                //             );

                for (auto i : {x, y}) 
                    for (auto j : {x, y}) 
                        stress[i][j].problem[k][l] = 
                            // deform[i][j];
                            coefficient[i][j][x][x][mat_id] * deform[x][x] +
                            coefficient[i][j][x][y][mat_id] * deform[x][y] +
                            coefficient[i][j][y][x][mat_id] * deform[y][x] +
                            coefficient[i][j][y][y][mat_id] * deform[y][y]
                            + coefficient[i][j][k][l][mat_id]
                            ;

                if ((k == z) and (l == z))
                    stress[z][z].problem[k][l] = 
                        coefficient[z][z][x][x][mat_id] * deform[x][x] +
                        coefficient[z][z][x][y][mat_id] * deform[x][y] +
                        coefficient[z][z][y][x][mat_id] * deform[y][x] +
                        coefficient[z][z][y][y][mat_id] * deform[y][y]
                        + coefficient[z][z][k][l][mat_id]
                        ;
            };

            for (auto i : {x, y}) 
                for (auto j : {x, y}) 
                    stress[j][z].problem[i][z] = 
                        problem_of_torsion_rod.flow[cell_num].second[j][i];


            for (auto i : {x, y, z}) 
                for (auto j : {x, y, z}) 
                    for (auto k : {x, y, z}) 
                        for (auto l : {x, y, z}) 
                        {
                            stress[j][i].problem[k][l] = 
                                stress[i][j].problem[k][l];
                            stress[i][j].problem[l][k] = 
                                stress[i][j].problem[k][l];
                            stress[j][i].problem[l][k] = 
                                stress[i][j].problem[k][l];
                            stress[k][l].problem[i][j] = 
                                stress[i][j].problem[k][l];
                        };

        };

        ++cell_num;

        dealii::Point<2> midle(
                (cell->vertex(0)[0] +
                 cell->vertex(1)[0] +
                 cell->vertex(2)[0] +
                 cell->vertex(3)[0]) / 4.0,
                (cell->vertex(0)[1] +
                 cell->vertex(1)[1] +
                 cell->vertex(2)[1] +
                 cell->vertex(3)[1]) / 4.0);

        cells_stress.push_back(std::make_pair(midle, stress));
        // printf("steress2 %f %f\n", 
        //         stress[y][z].problem[x][x],
        //         stress[x][x].problem[x][x]);
    };
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_meta_coefficients ()
{
    size_t len_vector_solution = this->domain.dof_handler.n_dofs();
    double mean_stress[dim + 1][dim + 1][dim + 1][dim + 1];

    std::array<std::array<std::array<std::array<
        dealii::Vector<double>, 3>, 3>, 3>, 3> 
        tau; 

    for (auto i : {x, y, z}) 
        for (auto j : {x, y, z}) 
            for (auto k : {x, y, z}) 
                for (auto l : {x, y, z}) 
                {
                    meta_coefficient[i][j][k][l] = 0.0;

                    FOR_N(0, cells_stress.size())
                    {
                        bool cell_exist = true;
                        for (auto bc : brocken_cell_in_line)
                            if (n == bc)
                                cell_exist = false;

                        if ((cell_exist)) 
                        meta_coefficient[i][j][k][l] += 
                            (cells_stress[n].second[i][j].problem[k][l] * cells_area[n]);
                    };

                    // double temp = 0.0;

                    // for (size_t o = 0; o < 
                    //         len_vector_solution
                    //         + len_vector_solution / 2
                    //         ; ++o)
                    // {
                    //     temp += 
                    //         human<Solution>(i, j, o) *
                    //         (-human<Stress>(k, l, o));
                    //     // if ((i == x) and (j == x) and (k == x) and (l == x)) 
                    //     //     printf("META %f %f %f\n",
                    //     //             human<Solution>(i, j, o),
                    //     //             (-human<Stress>(k, l, o)),
                    //     //              temp);
                    // };

                    // meta_coefficient[i][j][k][l] =
                    //     mean_coefficient[i][j][k][l] +
                    //     (temp / area_of_domain);
                    //     // if ((i == x) and (j == x) and (k == x) and (l == x)) 
                    //     //     printf("META %f %f\n", temp, area_of_domain);
                };

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::solve_system_equations ()
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
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::solve_system_equations_parallel (
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

    dealii::SolverControl solver_control (500000, 1e-8);

    dealii::SolverCG<> solver(solver_control);
   // Femenist::SolverSE<> solver(solver_control);

    this->system_equations.x = 0;

    solver .solve (
            this->system_equations.A,
            x,
            b
            ,dealii::PreconditionIdentity()
            );
//    );
    
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>:: print_stress (cst indx_i, cst indx_j)
{
    double meta[3][3][3][3] = {0.0};
    char suffix[3] = {'x', 'y', 'z'};
    FOR_I(0, dim + 1)
        FOR_J(0, dim + 1)
        {
            std::string file_name = "stress_";
            file_name += suffix[i];
            file_name += suffix[j];
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                fprintf(F, "%f %f  %f %f %f  %f %f %f  %f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        cells_stress[n].second[0][0].problem[i][j],
                        cells_stress[n].second[0][1].problem[i][j],
                        cells_stress[n].second[0][2].problem[i][j],
                        cells_stress[n].second[1][0].problem[i][j],
                        cells_stress[n].second[1][1].problem[i][j],
                        cells_stress[n].second[1][2].problem[i][j],
                        cells_stress[n].second[2][0].problem[i][j],
                        cells_stress[n].second[2][1].problem[i][j],
                        cells_stress[n].second[2][2].problem[i][j]);
                FOR_K(0, dim + 1)
                    FOR_L(0, dim + 1)
                    meta[i][j][k][l] += cells_stress[n].second[i][j].problem[k][l];
            };
            fclose(F);
        };

        {
            std::string file_name = "str_main_1";
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                bool cell_exist = true;
                for (auto i : brocken_cell_in_line)
                {
                    // printf("%d\n", i);
                    if (n == i)
                    {
                        cell_exist = false;
                    };
                };
                if ((material_of_cell[n] == 0) and (cell_exist))
                fprintf(F, "%f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        complete_main_stress_1[n]);
            };
            fclose(F);
        };

        {
            std::string file_name = "str_main_2";
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                fprintf(F, "%f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        complete_main_stress_2[n]);
            };
            fclose(F);
        };

        {
            std::string file_name = "str_yy";
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                fprintf(F, "%f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        complete_stress[n][1][1]);
            };
            fclose(F);
        };

        {
            std::string file_name = "str_xx";
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                fprintf(F, "%f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        complete_stress[n][0][0]);
            };
            fclose(F);
        };

        {
            std::string file_name = "str_xy";
            file_name += "_";
            file_name += std::to_string(indx_i);
            file_name += std::to_string(indx_j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_N(0, cells_stress.size())
            {
                fprintf(F, "%f %f %f\n", 
                        cells_stress[n].first(0),
                        cells_stress[n].first(1),
                        complete_stress[n][0][1]);
            };
            fclose(F);
        };

    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};
template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>:: output_results ()
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

    for (size_t i = 0; i < 9; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < 9; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (fabs(coefficient[im][in][jm][jn][0]) > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", fabs(coefficient[im][in][jm][jn][0]));
            else
                printf("%f   ", fabs(coefficient[im][in][jm][jn][0]));
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

            printf("\n");
    for (size_t i = 0; i < 9; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < 9; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (fabs(coefficient[im][in][jm][jn][1]) > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", fabs(coefficient[im][in][jm][jn][1]));
            else
                printf("%f   ", fabs(coefficient[im][in][jm][jn][1]));
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

            printf("\n");
    {
        // for (auto i : {x, y, z}) 
        //     for (auto j : {x, y, z}) 
        //         for (auto k : {x, y, z}) 
        //             for (auto l : {x, y, z}) 
        //                 FOR_N(0, cells_stress.size())
        //                     printf("2 TAU %f\n", cells_stress[n].second[i][j][k][l]);

        double meta[3][3][3][3] = {0.0};
        for (auto i : {x, y, z}) 
            for (auto j : {x, y, z}) 
                for (auto k : {x, y, z}) 
                    for (auto l : {x, y, z}) 
                        meta[i][j][k][l] = 0.0;

        char suffix[3] = {'x', 'y', 'z'};

        FOR_I(0, dim + 1) FOR_J(0, dim + 1) FOR_K(0, dim + 1) FOR_L(0, dim + 1)
        {
            FOR_N(0, cells_stress.size())
            {
                meta[i][j][k][l] += 
                    (cells_stress[n].second[i][j].problem[k][l] * cells_area[n]);
            };
            // meta[i][j][k][l] /= cells_stress.size();
        };
            // FOR_N(0, cells_stress.size())
            //     printf("ARREA %f\n", cells_stress[n].second[0][0].problem[0][0]);

    for (size_t i = 0; i < 9; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < 9; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (fabs(meta[im][in][jm][jn]) > 0.0000000001)
                printf("\x1B[31m%f\x1B[0m   ", fabs(meta[im][in][jm][jn]));
            else
                printf("%f   ", fabs(meta[im][in][jm][jn]));
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

            printf("\n");


    };

    FOR_I(0, brocken_cell.size())
        FOR_J(0, brocken_cell[i].size())
        {
            std::string file_name = "brocken_cells_";
            file_name += std::to_string(i);
            file_name += std::to_string(j);
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            for(st k : brocken_cell[i][j])
                fprintf(F, "%f %f %f\n", 
                        cells_stress[k].first(0),
                        cells_stress[k].first(1),
                        0.0);
            fclose(F);
        };

    FOR_I(0, brocken_cell.size())
    {
        std::string file_name = "brocken_cells_";
        file_name += std::to_string(i);
        file_name += ".gpd";
        FILE *F;
        F = fopen(file_name.data(), "w");
        FOR_J(0, brocken_cell[i].size())
            for(st k : brocken_cell[i][j])
                fprintf(F, "%f %f %f %f\n", 
                        cells_stress[k].first(0),
                        cells_stress[k].first(1),
                        0.0,
                        cells_area[k]);
        fclose(F);
    };

    FOR_I(0, brocken_cell.size())
        FOR_J(0, brocken_cell[i].size())
            for(st k : brocken_cell[i][j])
                brocken_cell_in_line_coor .push_back (cells_stress[k].first);
    FILE *F;
    F = fopen("brocken_cells_in_line.gpd", "w");
    fprintf(F, "%ld\n", brocken_cell_in_line.size());
    st count = 0;
    // FOR_I(0, brocken_cell.size())
    //     FOR_J(0, brocken_cell[i].size())
    //         for(st k : brocken_cell[i][j])
    FOR_I(0, brocken_cell_in_line.size())
    {
        fprintf(F, "%f %f %f %f %ld\n", 
                brocken_cell_in_line_coor[i](0),
                brocken_cell_in_line_coor[i](1),
                0.0,
                cells_area[brocken_cell_in_line[i]],
                brocken_cell_in_line[i]);
        ++count;
    };
    fclose(F);

    {
        std::string file_name = "max_stress.gpd";
        FILE *F;
        F = fopen(file_name.data(), "w");
        dbl gross_area = 0.0;
        fprintf(F, "%ld\n", max_complete_main_stress_1.size());
        // FOR_I(0, brocken_cell.size())
        // {
        //     FOR_J(0, brocken_cell[i].size())
        //         for(st k : brocken_cell[i][j])
        //             gross_area += cells_area[k];
        //     fprintf(F, "%f %f\n", 
        //             /* max_complete_main_stress_1[1] / */ max_complete_main_stress_1[i+1],
        //             gross_area);
        // };
        FOR_I(0, max_complete_main_stress_1.size())
            fprintf(F, "%f %f %f %f\n", 
                    max_complete_main_stress_1[i],
                    // gross_area,
                    fiz_coef[i][0],
                    fiz_coef[i][1],
                    fiz_coef[i][2]
                    );
        fclose(F);
    };


    REPORT_USE( 
            Report report;
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









