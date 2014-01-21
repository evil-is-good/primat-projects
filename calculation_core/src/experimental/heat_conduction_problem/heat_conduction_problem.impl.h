
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


                    |
                    |
                    |
                   \ /
                    '

template <u8 dim>
::M ()
{
    cu16 dofs_per_cell = finite_element.dofs_per_cell;

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell!=endc; ++cell)
    {
        dealii::FullMatrix<dbl> cell_matrix (dofs_per_cell, dofs_per_cell);

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
            for (size_t j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i,j) += this->m (i, j, cell); 
        };

        std::vector<u16> local_dof_indices (dofs_per_cell);
        cell ->get_dof_indices (local_dof_indices);

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
            for (size_t j = 0; j < dofs_per_cell; ++j)
               this->system_equations.A  .add (local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i,j));
        };
    };
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template <u8 dim>
::m ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_values   | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    cu16 n_q_points = quadrature_formula.size();

    fe_values .reinit (cell);

    dbl res = 0.0;

    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j < dim; ++j)
                if (i == j)
                {
                    res += this->coefficient[i][material_id] * 
                        fe_values.shape_grad (index_i, q_point)[i] *
                        fe_values.shape_grad (index_j, q_point)[i] *
                        fe_values.JxW(q_point);
                }
                else
                {
                    u8 n = i + j + dim - 1;
                    res += this->coefficient[n][material_id] * 
                        fe_values.shape_grad (index_i, q_point)[i] *
                        fe_values.shape_grad (index_j, q_point)[j] *
                        fe_values.JxW(q_point); 
                };
    return res;

};

template<u8 dim>
class ScalarProblemOnCell
{
    public:

    ScalarProblemOnCell ();

    {
        Domain<2> d;
        set_grid (d.grid);
        d.fe = FE_Q<2>(1);
        d.dof_init ();

        // dealii::SparseMatrix<dbl> matrix;
        SystemsLinearAlgebraicEquations slae;
        arr<dealii::Vector<dbl>, 2> solution;
        arr<dealii::Vector<dbl>, 2> rhsv;

        BlackOnWhiteSubstituter bows;
        prepare_system_equations_for_hcp_on_cell (slae, solution, rhsv, bows, d);

        Laplacian<2,1> element_matrix;

        HeatConductionProblemSup<dim>::TypeCoef coef;
        coef[x][x] .push_back (1.0);
        coef[y][y] .push_back (1.0);
        coef[x][y] .push_back (0.0);
        coef[x][x] .push_back (2.0);
        coef[y][y] .push_back (2.0);
        coef[x][y] .push_back (0.0);
        set_constants_for_hcp(element_matrix.C, coef);

        element_matrix.quadrature = dealii::QGauss<2>(2);
        element_matrix.init ();

        assemble_matrix<2,1,Laplasian<2,1>>(slae.matrix, element_matrix, bows);
        ASSEMBLER::assemble_matrix

        FOR(i, 0, 2)
        {
            arr<vec<dbl>, 2> coef_for_rhs;
            FOR(j, 0, 2)
                FOR(k, 0, element_matrix.C.size())
                {
                    coef_for_rhs[i] .puts_back (element_matrix.C[i][j][k]);
                };
            LoopBordersAsSource<2,1> element_rhsv (coef_for_rhs);

            assemble_rhsv_on_cell<2,1,LoopBordersAsSource<2,1>>(rhsv[i], element_rhsv);

            dealii::SolverControl solver_control (10000, 1e-12);
            dealii::SolverCG<> solver (solver_control);
            solver.solve (
                    slae.matrix,
                    solution[i],
                    rhsv[i]
                    ,dealii::PreconditionIdentity()
                    );
            FOR(j, 0, slae.solve[i].size())
                solution[i][j] = bows.subst (solution[i][j]);
        };

        FOR(i, 0, 2)
            print_sol_for_hcp(solution[i], d.dof_hendler, "name i");

        arr<dbl, 2> meta_coef;
        calculate_meta_coefficients (meta_coef, slae);

        printf(meta_coef);
    };
}

main()
{
    //HEAT_CONDUCTION_PROBLEM
    {
        Domain<2> d;
        set_grid (d.grid);
        d.fe = FE_Q<2>(1);
        d.dof_init ();

        SystemsLinearAlgebraicEquations slae;
        ATools::trivial_prepare_system_equations (slae, d);

        LaplacianScalar<2> element_matrix (d.fe);
        {
            arr<arr<vec<dbl>,2,2> coef;
            coef[x][x] .push_back (1.0);
            coef[y][y] .push_back (1.0);
            coef[x][y] .push_back (0.0);
            coef[y][x] .push_back (0.0);
            HCP::set_constants(element_matrix.C, coef);
        }

        Source<2,1> element_rhsv (func);

        Assembler::assemble_matrix<2> (slae.matrix, element_matrix, domain.dof_hendler);
        Assembler::assemble_rhsv<2> (slae.matrix, element_rhsv, domain.dof_hendler);

        std::vector<prmt::BoundaryValues<2,1>> bound (1);
        bound[0].function = [] (const dealii::Point<2> &p) {return p(0);};
        bound[0].boundari_indicator = 0;
        bound[0].type = Dirichlet;

        for (b : bound)
            b .apply_to (slae);
        // applay_boundary_values<2,1> (slae, bound);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );

        HCP::print_sol(slae.solution, d.dof_hendler, "name");
    };

    //ELASTIC_PROBLEM
    {
        Domain<2> d;
        set_grid (d.grid);
        d.fe = FESystem<2>(dealii::FE_Q<dim>(1), 2);
        d.dof_init ();

        SystemsLinearAlgebraicEquations slae;
        prepare_system_equations_for_hcp (slae, d);

        Laplacian<2,2> element_matrix;

        ElasticProblemSup<dim>::TypeCoef coef;
        Femenist::Function<std::array<double, dim>, dim> rhsv;

        FOR(i, 0, 2) FOR(j, 0, 2) FOR(k, 0, 2) FOR(l, 0, 2)
            coef[i][j][k][l] .resize (1);

        dbl lambda = 1.0;
        dbl mu     = 1.0;

        coef[0][0][0][0][0] = lambda + 2 * mu;
        coef[1][1][1][1][0] = lambda + 2 * mu;

        coef[0][0][1][1][0] = lambda;
        coef[1][1][0][0][0] = lambda;

        coef[0][1][0][1][0] = mu;
        coef[1][0][1][0][0] = mu;
        coef[0][1][1][0][0] = mu;
        coef[1][0][0][1][0] = mu;
        set_constants_for_elastic(element_matrix.C, coef);

        element_matrix.quadrature = dealii::QGauss<2>(2);
        element_matrix.init ();

        Source<2,2> element_rhsv (func);

        vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
        bound .push_back (boundary_value(
                    [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};},
                    0,
                    Neumann));

        applay_boundary_values<2,2> (slae, bound);

        assemble_matrix<2,2,Laplasian<2,2>>(slae.matrix, element_matrix);
        assemble_rhsv<2,2,Source<2,2>>(slae.rhsv, element_rhsv);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );

        print_sol_for_elastic(slae.solution, d.dof_hendler, "name");
    };

    //HEAT_CONDUCTION_PROBLEM_ON_CELL
    {
        set_grid (d.grid);

        HeatConductionProblemSup<dim>::TypeCoef coef;
        coef[x][x] .push_back (1.0);
        coef[y][y] .push_back (1.0);
        coef[x][y] .push_back (0.0);
        coef[x][x] .push_back (2.0);
        coef[y][y] .push_back (2.0);
        coef[x][y] .push_back (0.0);

        ScalarProblemOnCell problem(grid, coef);
        problem.solve();
        problem.print_result();
    };

    //ELASTIC_PROBLEM_ON_CELL
    {
        Domain<2> d;
        set_grid (d.grid);
        d.fe = FESystem<2>(dealii::FE_Q<dim>(1), 2);
        d.dof_init ();

        SystemsLinearAlgebraicEquationsOnCell<
            prmt::SymmetricTensor<dealii::Vector<dbl>, 2, 2>> slae;
        prepare_system_equations_for_ep_on_cell (slae, bows, d);

        Laplacian<2,2> element_matrix;

        ElasticProblemSup<dim>::TypeCoef coef;
        Femenist::Function<std::array<double, dim>, dim> rhsv;

        FOR(i, 0, 2) FOR(j, 0, 2) FOR(k, 0, 2) FOR(l, 0, 2)
            coef[i][j][k][l] .resize (2);
        {
            dbl lambda = [1.0, 2.0];
            dbl mu     = [1.0, 2.0];

            FOR(i, 0, 2)
            {
                coef[0][0][0][0][i] = lambda[i] + 2 * mu[i];
                coef[1][1][1][1][i] = lambda[i] + 2 * mu[i];

                coef[0][0][1][1][i] = lambda[i];
                coef[1][1][0][0][i] = lambda[i];

                coef[0][1][0][1][i] = mu[i];
                coef[1][0][1][0][i] = mu[i];
                coef[0][1][1][0][i] = mu[i];
                coef[1][0][0][1][i] = mu[i];
            };
        };
        set_constants_for_elastic(element_matrix.C, coef);

        element_matrix.quadrature = dealii::QGauss<2>(2);
        element_matrix.init ();

        Source<2,2> element_rhsv (func);

        vec<typename ElasticProblemSup<dim>::BoundaryValues > bound;
        bound .push_back (boundary_value(
                    [] (const dealii::Point<2> &p) {return arr<dbl, 2>{0.0, 0.0};},
                    0,
                    Neumann));

        applay_boundary_values<2,2> (slae, bound);

        assemble_matrix<2,2,Laplasian<2,2>>(slae.matrix, element_matrix);
        assemble_rhsv<2,2,Source<2,2>>(slae.rhsv, element_rhsv);

        dealii::SolverControl solver_control (10000, 1e-12);
        dealii::SolverCG<> solver (solver_control);
        solver.solve (
                slae.matrix,
                slae.solution,
                slae.rhsv
                ,dealii::PreconditionIdentity()
                );

        print_sol_for_elastic(slae.solution, d.dof_hendler, "name");
    };
}
