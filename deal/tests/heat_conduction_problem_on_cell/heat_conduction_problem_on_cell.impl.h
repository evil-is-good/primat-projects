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

#ifndef HEAT_CONDUCTION_PROBLEM_ON_CELL_IMPL

#define HEAT_CONDUCTION_PROBLEM_ON_CELL_IMPL

//#include "./heat_conduction_problem_on_cell.desc.h"

template <int n>
class Analit
{
    public:
        Analit (std::array<double, n> &distance, std::array<double, n> &coeff);
        double operator() (const double x);
    private:
        std::array<double, n + 1> coor;
        std::array<double, n> a;
        std::array<double, n> b;
        double C;
};

template <int n>
Analit<n>::Analit (std::array<double, n> &distance, std::array<double, n> &coeff)
{
    coor[0] = 0.0;
    FOR_I(0, n)
        coor[i + 1] = coor[i] + distance[i];

    double L = 0.0;
    for (auto i : distance)
        L += i;

    double ll = 0.0;
    FOR_I(0, n)
        ll += distance[i] / coeff[i];

    C = L / ll;

    FOR_I(0, n)
        a[i] = C / coeff[i] - 1.0;

    b[0] = 0.0;
    b[n - 1] = - a[n - 1];

    for (size_t i = n - 2; i > 0; --i)  
    {
        b[i] = a[i] * (- coor[i + 1]) + a[i + 1] * coor[i + 1] + b[i + 1]; 
        printf("%f %f %f %f %f\n", a[i], coor[i], a[i + 1], coor[i], b[i + 1]);
    };

    printf("C=%f\n", C);
    for (auto i : coor)
        printf("coor=%f\n", i);
    for (auto i : a)
        printf("a=%f\n", i);
    for (auto i : b)
        printf("b=%f\n", i);
    for (auto i : distance)
        printf("dis=%f\n", i);
    for (auto i : coeff)
            printf("coeff=%f\n", i);

};

template <int n>
double Analit<n>::operator() (const double x)
{
    FOR_I(0, n)
        if ((x > coor[i] - 1e-10) and (x < coor[i + 1] + 1e-10))
        {
//            printf("%ld %f %f %f %f\n", i, a[i], x, b[i], (a[i] * x + b[i]));
            return a[i] * x + b[i];
        };

    // printf("\x1B[31mERROR IN Analit x = %f\x1B[0m\n", x);
    return 0.0;
};

template<uint8_t dim>
ElementRightHandSideVectorHeatConductionProblemOnCell<dim>::
ElementRightHandSideVectorHeatConductionProblemOnCell ()
    :
        ElementRightHandSideVector<
            dim, 
            double, 
            std::array<std::vector<double>, dim> > ()
//            std::array<Femenist::Function<double,dim>,dim> > ()
{

};

template<uint8_t dim>
void ElementRightHandSideVectorHeatConductionProblemOnCell<dim> :: 
set_coefficient (const std::array<std::vector<double>, dim> &coef)
//        std::array<Femenist::Function<double,dim>,dim>  &coef)
{
//    this->coefficient .swap (coef);
    for (size_t i = 0; i < dim; ++i)
    {
        this->coefficient[i] .clear ();

        for (size_t j = 0; j < coef[i].size(); ++j)
            this->coefficient[i] .push_back (coef[i][j]);
    };
};

template<uint8_t dim>
double ElementRightHandSideVectorHeatConductionProblemOnCell<dim> :: 
operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0;

//    printf("Qadrature: index=%ld\n", index_i);
    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
    {
        for(size_t i = 0; i < dim; ++i)
        {//this->coefficient[i](fe_values.quadrature_point(q_point)) *
            res += -fe_values.shape_grad (index_i, q_point)[i] *
                this->coefficient[i][material_id] *
                    fe_values.JxW(q_point);

//            printf("coor_%ld=%f,",fe_values.quadrature_point(q_point)[i]);
//            printf(" coef = %f,", 
//                    this->coefficient[i][material_id]);
//            printf(" grad = %f\n", fe_values.shape_grad (index_i,q_point )[i]);
        };
    };
//    printf("\n");
//    printf("%ld %f\n", index_i, res);

    return res;
};

template<uint8_t dim>
HeatConductionProblemOnCell<dim>::HeatConductionProblemOnCell (
        const dealii::Triangulation<dim> &triangulation,
        const ContainerCoef &coef)
:
    Problem< 
        dim,
        ElementStiffnessMatrixLaplaceProblem<dim>,
        ElementRightHandSideVectorHeatConductionProblemOnCell<dim> > (),
    finite_element (1)
{
    this->element_stiffness_matrix .set_coefficient (coef);

//    this->coefficient .swap (coef);
    for (size_t i = 0; i < num_coef; ++i)
        for (size_t j = 0; j < coef[i].size(); ++j)
        this->coefficient[i] .push_back (coef[i][j]);

    this->domain.triangulation .copy_triangulation (triangulation);
};

template<uint8_t dim>
HeatConductionProblemOnCell<dim>::~HeatConductionProblemOnCell ()
{
    this->domain .clean ();
};

template<uint8_t dim>
uint8_t HeatConductionProblemOnCell<dim>::conver (uint8_t index_i, uint8_t index_j)
{
    if (index_i == index_j)
        return index_i;
    else
        return (index_i + index_j + dim - 1);
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::solved ()
{
//    printf("AAA 1\n");
    REPORT setup_system ();
//    printf("AAA 2\n");
    REPORT assemble_matrix_of_system ();
    {
        FILE *F;
        F = fopen("matrix_cell.gpd","w");
        for (size_t i = 0; i < this->system_equations.A.m(); ++i)
            for (size_t j = 0; j < this->system_equations.A.n(); ++j)
                if (this->system_equations.A.el(i,j))
                    fprintf(F,"%ld %ld %f\n", i, j, this->system_equations.A.el(i,j));
        fclose(F);
    };
//    printf("AAA 3\n");
    REPORT calculate_mean_coefficients ();
    REPORT calculate_cells_area ();
//    printf("%f, %f, %f\n", mean_coefficient[0],
//                           mean_coefficient[1],
//                           mean_coefficient[2]);
//    printf("AAA 4\n");

    std::array<std::vector<double>, dim> coef_for_rhs;
    for(size_t i = 0; i < dim; ++i)
        coef_for_rhs[i] .resize (coefficient[i].size());

    for(size_t i = 0; i < dim; ++i)
    {
        for(size_t j = 0; j < dim; ++j)
        {
            for(size_t k = 0; k < coefficient[conver(i,j)].size(); ++k)
                coef_for_rhs[j][k] = coefficient[conver(i,j)][k];
        };
        
        solution[i]  = 0;
        heat_flow[i] = 0;

        this->element_rh_vector .set_coefficient (coef_for_rhs);

        REPORT assemble_right_vector_of_system_parallel (assigned_to heat_flow[i]);
    };

//    {
//        heat_flow[0] = 0.0;
//        FOR_I (0, this->system_equations.A.m())
//            FOR_J (0, this->system_equations.A.n())
//            heat_flow[0](i) += this->system_equations.A.el(i,j) * anal_x(j);
//        double temp = 0.0;
//        FOR_I(0, this->system_equations.A.m())
//            temp += heat_flow[0](i);
//        printf("FFFFFFFFFFFFFF %f\n", temp);
//    };

    {
        FILE *F;
        F = fopen("A.gpd","w");
        for (size_t i = 0; i < this->system_equations.A.m(); ++i)
            if (not black_on_white_substituter.is_black(i))
            {
                for (size_t j = 0; j < this->system_equations.A.n(); ++j)
                    if (not black_on_white_substituter.is_black(j))
                        if (this->system_equations.A .el (i,j))
                        fprintf(F,"%f %f %f\n", i*1., j*1., this->system_equations.A .el (i,j));
//                        fprintf(F,"%f ", this->system_equations.A .el (i,j));
//                fprintf(F,"%f\n", this->system_equations.A.el(i,this->system_equations.A.n()-1));
            };
        fclose(F);
    };
    {
        FILE *F;
        F = fopen("b.gpd","w");
        for (size_t i = 0; i < this->system_equations.b.size(); ++i)
                if (not black_on_white_substituter.is_black(i))
                    fprintf(F,"%f\n", heat_flow[0](i));
        fclose(F);
    };
//    {
//        FILE *F;
//        F = fopen("specA.gpd","w");
//        for (size_t i = 0; i < this->system_equations.b.size(); ++i)
//                if (not black_on_white_substituter.is_black(i))
//                    fprintf(F,"%f\n", this->system_equations.A .el (22,i));
//        fclose(F);
//    };

//    omp_set_dynamic(0);     
//    omp_set_num_threads(2); 
//#pragma omp parallel default(shared)
//    {
//#pragma omp sections
//        {
//#pragma omp section
//            {
//                REPORT solve_system_equations_parallel (
//                        solution[0], heat_flow[0]);
//
//                for (size_t j = 0; j < this->system_equations.x.size(); ++j)
//                    this->system_equations.x(j) = this->system_equations.x(
//                            black_on_white_substituter .subst (j));
//
////                solution[0] = this->system_equations.x;
////
////                heat_flow[0] = this->system_equations.b;
//            }
//#pragma omp section
//            {
//                REPORT solve_system_equations_parallel (
//                        solution[1], heat_flow[1]);
//
//                for (size_t j = 0; j < this->system_equations.x.size(); ++j)
//                    this->system_equations.x(j) = this->system_equations.x(
//                            black_on_white_substituter .subst (j));
//
////                solution[1] = this->system_equations.x;
////
////                heat_flow[1] = this->system_equations.b;
//            }
//        }
//    }
#pragma omp parallel for
    for(size_t i = 0; i < dim; ++i)
    {
        REPORT solve_system_equations_parallel (
                solution[i], heat_flow[i]);

        for (size_t j = 0; j < this->system_equations.x.size(); ++j)
            solution[i](j) = solution[i](
                    black_on_white_substituter .subst (j));
    };

    {
        FILE *F;
        F = fopen("x.gpd","w");
        for (size_t i = 0; i < this->system_equations.x.size(); ++i)
                if (not black_on_white_substituter.is_black(i))
            fprintf(F,"%f\n", solution[0](i));
        fclose(F);
    };
//#pragma omp parallel for
//    for(size_t i = 0; i < dim; ++i)
//    {
//        for(size_t j = 0; j < dim; ++j)
//        {
//            for(size_t k = 0; k < coefficient[conver(i,j)].size(); ++k)
//                coef_for_rhs[j][k] = coefficient[conver(i,j)][k];
//        };
//        
//        this->system_equations.x = 0;
//        this->system_equations.b = 0;
//
//        this->element_rh_vector .set_coefficient (coef_for_rhs);
//
//        REPORT assemble_right_vector_of_system ();
//
//        REPORT solve_system_equations ();
//
//        for (size_t j = 0; j < this->system_equations.x.size(); ++j)
//            this->system_equations.x(j) = this->system_equations.x(
//                    black_on_white_substituter .subst (j));
//
//        solution[i] = this->system_equations.x;
//
//        heat_flow[i] = this->system_equations.b;
//    };

//    printf("AAA 5\n");
    REPORT calculate_meta_coefficients ();
    REPORT calculate_flow ();
//    printf("AAA 6\n");

//    printf("bbbbbbbbbbbbbbbbbbbbbbbbbb %f %f\n", heat_flow[0](21), heat_flow[0](22));
    REPORT_USE( 
            prmt::Report report;
            report.result = _report .result;
            _return (report););
};

template<uint8_t dim>
void HeatConductionProblemOnCell<dim>::print_result (const std::string &file_name)
{
    output_file_name = file_name;
    output_results ();
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::setup_system ()
{
    this->domain.dof_handler.distribute_dofs (finite_element);

    dealii::CompressedSparsityPattern c_sparsity (
            this->domain.dof_handler.n_dofs());

    dealii::DoFTools ::make_sparsity_pattern (
            this->domain.dof_handler, c_sparsity);

    {
    std::ofstream out ("sparsity_pattern.1");
    c_sparsity.print_gnuplot (out);
    }

    DomainLooper<dim, 0> dl;
    REPORT dl .loop_domain(
            this->domain.dof_handler,
            black_on_white_substituter,
            c_sparsity);

    {
    std::ofstream out ("sparsity_pattern.2");
    c_sparsity.print_gnuplot (out);
    }

    c_sparsity.compress ();

    this->system_equations .reinit (c_sparsity, 
            this->domain.dof_handler.n_dofs());
    
    for (size_t i = 0; i < dim; ++i)
    {
        solution[i]  .reinit (this->domain.dof_handler.n_dofs());
        heat_flow[i] .reinit (this->domain.dof_handler.n_dofs());
    };
    anal_x .reinit (this->domain.dof_handler.n_dofs()); ////////////////
//    printf("AAAAAAAAAAAAAAAAA\n");

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::calculate_flow ()
{
    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients | 
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const uint8_t dofs_per_cell = finite_element.dofs_per_cell;
    const uint8_t num_quad_points = quadrature_formula.size();

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();
    using cell_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    // auto summ_of = [num_quad_points, &fe_values, this] (st num_dofs_on_cell){
    //     auto integrals_on = [&] (cell_iterator c){
    //         auto of_grad_funcions_from = [&] (st problem_num){

    //             arr<dbl, 2> grad = {0.0};

    //             FOR(differencial, 0, dim)
    //             {
    //                 FOR(function, 0, num_dofs_on_cell)
    //                 {
    //                     dbl integ_on_cell = 0.0;
    //                     FOR (q_point, 0, num_quad_points)
    //                     {
    //                         integ_on_cell += 
    //                             fe_values.shape_grad (function, q_point)[differencial] * 
    //                             fe_values.JxW(q_point);
    //                     };
    //                     grad[differencial] += 
    //                         solution[problem_num](c->vertex_dof_index(function, 0)) * 
    //                         integ_on_cell;
    //                 };
    //             };
    //             return grad;
    //         };
    //         return of_grad_function_fromi();
    //     };
    //     return integrals_on;
    // };

    // auto mean_grad_on_cell = [dofs_per_cell, &cell, &fe_values, this] (st problem_num){
    //     arr<dbl, 2> grad = {0,0}
    //     FOR(differencial, 0, dim)
    //         grad[differencial] = 
    // };


    st cell_num = 0;
    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);

        std::array<std::array<double, dim>, dim> tau = {0.0};
        bool cell_exist = true;
        for (auto bc : brocken_cell_in_line)
            if (cell_num == bc)
                cell_exist = false;

        if (cell_exist)
        {

        // FOR_I(0, dim) FOR_J(0, dim) tau[i][j] = 0.0;

        double area = 0.0;
        FOR_O(0, num_quad_points)
            area += fe_values.JxW(o);

        // struct {arr<dbl, 2> mean_grad_on_cell;} problem[2];

        // FOR (problem_num, 0, dim)
        // {
        //     problem[problem_num].mean_grad_on_cell = 
        //     summ_of(dofs_per_cell).integrals_on(cell).of_grad_functions_from(problem_num) /
        //     area;
        // };

        // dbl grad[2][2] = {0};
        struct {dealii::Tensor<1, dim> mean_grad_on_cell;} problem[2];
        FOR_I(0, dim)
        {
            // FOR_J(0, dim)
            // {
                // grad[i][j] = 0.0;
                FOR_N(0, dofs_per_cell)
                {
                    // dbl temp = 0.0;
                    dealii::Tensor<1, dim> temp;
                    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
                    {
                        temp += 
                            fe_values.shape_grad (n, q_point) * //[j] * 
                            // solution[i](cell->vertex_dof_index(n, 0)) *
                            // this->coefficient[conver(j, k)][cell->material_id()] *
                            fe_values.JxW(q_point);
                    };
                    problem[i].mean_grad_on_cell  += 
                        (solution[i](cell->vertex_dof_index(n, 0)) * temp);
                };
                problem[i].mean_grad_on_cell /= area;
            // };
        };

        FOR_I(0, dim) FOR_J(0, dim)
            tau[i][j] += 
            this->coefficient[conver(i, 0)][cell->material_id()] * problem[j].mean_grad_on_cell[0] +
            this->coefficient[conver(i, 1)][cell->material_id()] * problem[j].mean_grad_on_cell[1] +
            this->coefficient[conver(i, j)][cell->material_id()];
        };
        ++cell_num;
            // printf("tau %f\n", tau[i][j]);
            // printf("area %f\n", area);
            // printf("taunew %f\n", tau[i][j]);

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

        flow.push_back(std::make_pair(midle, tau));
    };
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::assemble_matrix_of_system ()
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

//        for (size_t i = 0; i < dofs_per_cell; ++i)
//            for (size_t j = 0; j < dofs_per_cell; ++j)
//                cell_matrix(i,j) += this->element_stiffness_matrix
//                    (i, j, quadrature_formula, fe_values, cell->material_id()); 
        for (unsigned int q_point=0; q_point<4; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                            fe_values.shape_grad (j, q_point) *
                            fe_values.JxW (q_point)
                    * this->coefficient[0][cell->material_id()]
                        );
            }

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
prmt::Report HeatConductionProblemOnCell<dim>::assemble_right_vector_of_system ()
{
    dealii::QGauss<dim> quadrature_formula(2);

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
//                printf("CCCCCCCC %f\n", cell_rhs(i));
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
dealii::Point<dim, double> get_grad (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        uint8_t index_vertex)
{
    dealii::Point<dim, double> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    double f1 = index_vertex == 0 ? 1.0 : 0.0;
    double f2 = index_vertex == 1 ? 1.0 : 0.0;
    double f3 = index_vertex == 2 ? 1.0 : 0.0;
    double f4 = index_vertex == 3 ? 1.0 : 0.0;

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    

    grad(0) = b + d * cell->vertex(index_vertex)(1);
    grad(1) = c + d * cell->vertex(index_vertex)(0);

    return grad;
};

template<uint8_t dim>
bool contains (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const typename dealii::Point<dim, double> &p)
{
    const dealii::Point<dim, double> p1 = cell->vertex(0);
    const dealii::Point<dim, double> p2 = cell->vertex(1);
    const dealii::Point<dim, double> p3 = cell->vertex(2);
    const dealii::Point<dim, double> p4 = cell->vertex(3);

    bool res = false;

    auto above_the_line = 
        [p] (const dealii::Point<dim, double> pl, 
                const dealii::Point<dim, double> pr) -> bool
        {
            const uint8_t x = 0;
            const uint8_t y = 1;

            if ((
                        pl(x) * pr(y) - 
                        pr(x) * pl(y) + 
                        p(x)  * (pl(y) - pr(y)) + 
                        p(y)  * (pr(x) - pl(x))) >= -1e-12)
                return true;
            else
                return false;
        };

    if (above_the_line (p1, p4))
    {
        if (above_the_line (p4, p3) and above_the_line (p3, p1))
            res = true;
    }
    else
    {
        if (above_the_line (p1, p2) and above_the_line (p2, p4))
            res = true;
    };

    return res;
};

template<uint8_t dim>
dealii::Point<dim, double> integral (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        uint8_t index_vertex,
        double dop)
{
    dealii::Point<dim, double> res;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(3)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(1)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(3)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(1)(1);

    double f1 = index_vertex == 0 ? 1.0 : 0.0;
    double f2 = index_vertex == 3 ? 1.0 : 0.0;
    double f3 = index_vertex == 2 ? 1.0 : 0.0;
    double f4 = index_vertex == 1 ? 1.0 : 0.0;

//    double x1 = cell->vertex(0)(0);
//    double x2 = cell->vertex(1)(0);
//    double x3 = cell->vertex(2)(0);
//    double x4 = cell->vertex(3)(0);
//
//    double y1 = cell->vertex(0)(1);
//    double y2 = cell->vertex(1)(1);
//    double y3 = cell->vertex(2)(1);
//    double y4 = cell->vertex(3)(1);
//
//    double f1 = index_vertex == 0 ? 1.0 : 0.0;
//    double f2 = index_vertex == 1 ? 1.0 : 0.0;
//    double f3 = index_vertex == 2 ? 1.0 : 0.0;
//    double f4 = index_vertex == 3 ? 1.0 : 0.0;

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

    double A1 = (y1 - y2);
    double B1 = (x2 - x1);
    double A2 = (y3 - y4);
    double B2 = (x4 - x3);
    double C1 = (x2 * y1 - x1 * y2);
    double C2 = (x4 * y3 - x3 * y4); // Ax+By=C
    
    double l  = pow(pow(A2, 2.0) + pow(B2, 2.0), 0.5);
    double h1 = fabs(x1 * A2 + y1 * B2 - C2) / l;
    double h2 = fabs(x2 * A2 + y2 * B2 - C2) / l;
    double S = (h1 + h2) * l / 2.0;

    double x0 = (C2 - B2 / B1 * C1) / (A2 - B2 / B1 * A1);
    double y0 = (C2 - A2 / A1 * C1) / (B2 - A2 / A1 * B1);

//    double xv1 = x1;
//    double xv2 = x4;
//    double xv4 = x3;
//    double xv3 = x2;
//
//    double yv1 = y1;
//    double yv2 = y4;
//    double yv4 = y3;
//    double yv3 = y2;
//
//    double fv1 = f1;
//    double fv2 = f4;
//    double fv4 = f3;
//    double fv3 = f2;
//
//    double A1 = (yv1 - yv2);
//    double B1 = (xv2 - xv1);
//    double A2 = (yv3 - yv4);
//    double B2 = (xv4 - xv3);
//    double C1 = (xv2 * yv1 - xv1 * yv2);
//    double C2 = (xv4 * yv3 - xv3 * yv4); // Ax+By=C
//    
//    double l  = pow(pow(A2, 2.0) + pow(B2, 2.0), 0.5);
//    double h1 = fabs(xv1 * A2 + yv1 * B2 - C2) / l;
//    double h2 = fabs(xv2 * A2 + yv2 * B2 - C2) / l;
//    double S = (h1 + h2) * l / 2.0;
//
//    double x0 = (C2 - B2 / B1 * C1) / (A2 - B2 / B1 * A1);
//    double y0 = (C2 - A2 / A1 * C1) / (B2 - A2 / A1 * B1);

//    printf("%f %f %f %f %f\n", y1, y2, y3, y4, y0);

//    double y0 = 0.0;
//    {
//        double m1 = x4 * y3 - x3 * y4;
//        double m2 = x2 * y1 - x1 * y2;
//        double m3 = x4 - x3;
//        double m4 = x2 - x1;
//        double m5 = (y3 - y4) / (y1 - y2);
//        y0 = (m1 - m2 * m5) / (m3 - m4 * m5);
//    };

    FILE *F;
    F = fopen("points.gpd", "a");
//    srand (time(NULL));
//    res(0) = 0.0;
//    FOR_I(0, 10000)
//    {
//        double xr = x0 + (0.2 - (0.4 / (rand() % 1000 + 1)));
//        double yr = y0 + (0.2 - (0.4 / (rand() % 1000 + 1)));
//        if (contains<dim>(cell, dealii::Point<dim>(xr, yr)))
//            res(0) += (b + d * yr);
//        fprintf(F,"%f %f 0.0\n", xr, yr);
//    };
//
//    res(0) /= 10000;
//
//    res(0) = S * res(0);
// 
    y0 = (y1 + y2 + y3 + y4) / 4.0;
    x0 = (x1 + x2 + x3 + x4) / 4.0;
//    y0 = (dop / S - b) / d;
    fprintf(F,"%f %f 0.0\n", x0, y0);
    fclose(F);

    res(0) = S * (b + d * y0);
    res(1) = 0.0;

    return res;
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::calculate_cells_area ()
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
        // printf("AREA %f\n", area);
    };

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>
::assemble_right_vector_of_system_parallel (dealii::Vector<double>& b)
{
    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients | 
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

    dealii::Vector<double> cell_rhs (dofs_per_cell);
    dealii::Vector<double> cell_x (dofs_per_cell); //////
    anal_x = 0.0;

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    {
        FILE *F;
        F = fopen("points.gpd", "w");
        fclose(F);
    };

//    std::array<double, 3> l1 = {1./3., 1./3., 1./3.}; 
//    std::array<double, 3> l2 = {1., 10., 1.}; 
//    Analit<3> analit(l1, l2);
    std::array<double, 2> l1 = {0.2, 0.8}; 
    std::array<double, 2> l2 = {10., 1.}; 
    Analit<2> analit(l1, l2);
    double sum9  = 0.0;
    double sum13 = 0.0;
    size_t ct = 0;
    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);
        cell_rhs = 0;
        cell_x = 0.0; //

        // printf("bbb ");
        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
                cell_rhs(i) += 
//                    integral<dim>(cell, i, 1.0)[0]
//                    * this->coefficient[0][cell->material_id()]
                    this->element_rh_vector
                    (i, quadrature_formula, fe_values, cell->material_id()); 
        ;
                cell_x(i) = analit(cell->vertex(i)(0));
                // printf("%f ", cell_rhs(i));
        };
        // printf("\n");

        cell ->get_dof_indices (local_dof_indices);

        for (size_t i = 0; i < dofs_per_cell; ++i)
            local_dof_indices[i] = black_on_white_substituter .subst (
                    local_dof_indices[i]);

//        if (ct == 5)
//        {
//            integral<dim>(cell, 0);
//        };
//            printf("%ld %f %f %f %f %d %d %d %d %f %f %f %f\n", 
//                    ct,
//                    cell_rhs(0),
//                    cell_rhs(1),
//                    cell_rhs(2),
//                    cell_rhs(3),
//                local_dof_indices[0],
//                local_dof_indices[1],
//                local_dof_indices[2],
//                local_dof_indices[3],
//                integral<dim>(cell, 0, cell_rhs(0))[0],
//                integral<dim>(cell, 1, cell_rhs(1))[0],
//                integral<dim>(cell, 2, cell_rhs(2))[0],
//                integral<dim>(cell, 3, cell_rhs(3))[0]
//                    );
            {
                uint8_t q_point = 0;
                dealii::Point<dim, double> p(fe_values.quadrature_point(q_point));
                uint8_t index_vertex = 0;

                double x1 = 0.0;//cell->vertex(0)(0);
                double x2 = 1.0;//cell->vertex(1)(0);
                double x3 = 0.0;//cell->vertex(2)(0);
                double x4 = 1.0;//cell->vertex(3)(0);

                double y1 = 0.0;//cell->vertex(0)(1);
                double y2 = 0.0;//cell->vertex(1)(1);
                double y3 = 1.0;//cell->vertex(2)(1);
                double y4 = 1.0;//cell->vertex(3)(1);

                double f1 = index_vertex == 0 ? 1.0 : 0.0;
                double f2 = index_vertex == 1 ? 1.0 : 0.0;
                double f3 = index_vertex == 2 ? 1.0 : 0.0;
                double f4 = index_vertex == 3 ? 1.0 : 0.0;

                double a=-(x2*y2*x3*y4*f1+y1*x3*x4*y4*f2-y1*x3*x2*y2*f4-x1*y1*x3*y4*f2-x1*y3*x4*y4*f2-x1*f3*x2*y2*y4+x1*x3*y3*f2*y4+x1*y3*x2*y2*f4+y1*f3*x4*x2*y2-y3*x4*x2*y2*f1-y3*y1*x3*x4*f2+x1*y3*x4*y1*f2-x2*y3*x3*y4*f1+x1*y1*f3*y4*x2+y1*y3*x3*x2*f4+x2*y3*x4*y4*f1-y1*f3*y4*x4*x2+x3*y3*x4*y2*f1-x3*f1*y2*x4*y4+y2*x3*x1*y1*f4+y2*x1*f3*y4*x4-y2*x4*x1*y1*f3-y2*x1*y3*x3*f4-x1*y1*y3*x2*f4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
                double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
                double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
                double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

                double val = a + b * p(0) + c * p(1) + d * p(0) * p(1);

//                printf("%f %f %f %f %f\n", p(0), p(1),
//                        fe_values.shape_grad (index_vertex, q_point)[0]
////                        *fe_values.jacobian(q_point)[0][0]
//                        , p(1) - 1
//                        , b + d * p(1)
//                        );
            };

//            const uint8_t num_quad_points = quadrature_formula.size();
//
//            double res = 0;
//
//            for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
//            {
//                for(size_t i = 0; i < dim; ++i)
//                {
//                    res += -fe_values.shape_grad (0, q_point)[i] *
////                        this->coefficient[i][material_id] *
//                        fe_values.JxW(q_point);
//                };
//                printf("%f\n", fe_values.shape_grad (0, q_point)[0]);
//            };
//        };

        for (size_t i = 0; i < dofs_per_cell; ++i)
        {
//            if (local_dof_indices[i] == 3)
//                printf("CCCCCCCC %d %d %f\n", i, local_dof_indices[i], cell_rhs(i));
            b (local_dof_indices[i]) +=
                cell_rhs (i);
            anal_x (local_dof_indices[i]) =
                cell_x (i);
        };

//        printf("%ld || %d %d %d %d || %d %d %d %d\n", ct, 
//                cell->vertex_dof_index(0,0),
//                cell->vertex_dof_index(1,0),
//                cell->vertex_dof_index(2,0),
//                cell->vertex_dof_index(3,0),
//                local_dof_indices[0],
//                local_dof_indices[1],
//                local_dof_indices[2],
//                local_dof_indices[3]
//                );

//        if (
//                (ct == 3)  or
//                (ct == 5)  or
//                (ct == 34) or
//                (ct == 35))
//        {
//            size_t a = 0;
//
//            FOR_I(0, 4)
//                if (local_dof_indices[i] == 9)
//                    a = i;
//
//            printf("9 %ld %f\n", ct, cell_rhs (a));
//
//            sum9 += cell_rhs (a);
//        };
//
//        if (
//                (ct == 5)  or
//                (ct == 9)  or
//                (ct == 22)  or
//                (ct == 29)  or
//                (ct == 32) or
//                (ct == 35))
//        {
//            size_t a = 0;
//
//            FOR_I(0, 4)
//                if (local_dof_indices[i] == 13)
//                    a = i;
//
//            printf("13 %ld %f\n", ct, cell_rhs (a));
//
//            sum13 += cell_rhs (a);
//        };
//
//        if (ct == 5)
//            printf("%f %f %f %f %d %d %d %d\n", 
//                    cell_rhs(0),
//                    cell_rhs(1),
//                    cell_rhs(2),
//                    cell_rhs(3),
//                local_dof_indices[0],
//                local_dof_indices[1],
//                local_dof_indices[2],
//                local_dof_indices[3]
//                    );
        ct++;
    };

    printf("%f %f\n", sum9, sum13);

//    for(auto i : b)
//        printf("!!!!!!!%f\n", i);
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::calculate_mean_coefficients ()
{
    dealii::QGauss<dim>  quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    for (size_t i = 0; i < num_coef; ++i)
        mean_coefficient[i] = 0.0;
    area_of_domain = 0.0;

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values .reinit (cell);

        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
            for (size_t i = 0; i < num_coef; ++i)
                mean_coefficient[i] += 
                    this->coefficient[i][cell->material_id()] *
                    fe_values.JxW(q_point);
//                    this->coefficient[i](fe_values.quadrature_point(q_point)) *

        for (size_t q_point = 0; q_point < n_q_points; ++q_point)
            area_of_domain += fe_values.JxW(q_point);
    };

    for (size_t i = 0; i < num_coef; ++i)
        mean_coefficient[i] /= area_of_domain;
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::calculate_meta_coefficients ()
{
    size_t len_vector_solution = this->domain.dof_handler.n_dofs();
    double mean_heat_flow[dim][dim];

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
        {
            mean_heat_flow[i][j] = 0.0;

            for (size_t k = 0; k < len_vector_solution; ++k)
                mean_heat_flow[i][j] += solution[i](k) * (-heat_flow[j](k));

            meta_coefficient[conver(i,j)] = mean_coefficient[conver(i,j)] +
                // 0.0;
                mean_heat_flow[i][j] / area_of_domain;
        };

    // FOR_I(0, dim) FOR_J(0, dim) 
    // {
    //     meta_coefficient[conver(i,j)] = 0.0;
    //     FOR_N(0, flow.size())
    //     {
    //         bool cell_exist = true;
    //         // for (auto bc : brocken_cell_in_line)
    //         //     if (n == bc)
    //         //         cell_exist = false;

    //         if (cell_exist)
    //             meta_coefficient[conver(i,j)] += (flow[n].second[i][j] * cells_area[n]);
    //     };
    //     // meta[i][j] /= flow.size();
    // };
   
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::solve_system_equations ()
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

//    {
//        dealii::SolverControl pred_solver_control (100000, 1e-8);
//
////        Femenist::SolverSE<> pred_solver(pred_solver_control);
//        dealii::SolverCG<> pred_solver(pred_solver_control);
//
//        this->system_equations.x = 0;
//
//        pred_solver .solve (
//                this->system_equations.A,
//                this->system_equations.x,
//                this->system_equations.b
////                );
//             ,dealii::PreconditionIdentity());
//    };
    this->system_equations.x = 0;

    dealii::SolverControl solver_control (100000, 1e-8);

//    Femenist::SolverSE<> solver(solver_control);

    dealii::SolverCG<> solver(solver_control);
//    dealii::SolverRelaxation<> solver(solver_control);
//    dealii::SolverMinRes<>     solver(solver_control);
//    dealii::SolverGMRES<>     solver(solver_control);

//    dealii::RelaxationBlockSOR<dealii::SparseMatrix<double> > relax;
//    relax.initialize(this->system_equations.A, 0.5);
//    dealii::PreconditionJacobi<> preconditioner;
//    preconditioner.initialize(this->system_equations.A, 1.0);

//    this->system_equations.x = 0;

//    REPORT solver .solve (
//            this->system_equations.A,
//            this->system_equations.x,
//            this->system_equations.b);
    
    solver .solve (
            this->system_equations.A,
            this->system_equations.x,
            this->system_equations.b
//            );
//            ,preconditioner);
//            ,relax);
             ,dealii::PreconditionIdentity());
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>::solve_system_equations_parallel (
                dealii::Vector<double>& x, dealii::Vector<double>& b)
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

    x = 0;
//    FOR_I (0, x.size())
////        x(i) = this->system_equations.A .el (i, 12);
//        x(i) = i * 1.0;

//    std::array<double, 3> l1 = {1./3., 1./3., 1./3.}; 
//    std::array<double, 3> l2 = {1., 10., 1.}; 
//    Analit<3> analit(l1, l2);
//    double temp = 0.0;
//    for (size_t i = 0; i < this->system_equations.b.size(); ++i)
//    {
//        temp +=
//                this->system_equations.A .el (3,i) *
//                anal_x(i);
//        printf("AAAA %ld %f %f %f %f\n", 
//                i,
//                this->system_equations.A .el (3,i),
//                anal_x(i),
//                this->system_equations.A .el (3,i) *
//                anal_x(i),
//                temp
//                );
////        if (not black_on_white_substituter.is_black(i))
//            x(i) = this->system_equations.A .el (3,i);
//            printf("%ld %f\n", i, this->system_equations.A .el (3,i));
//    }

    dealii::SolverControl solver_control (50000, 1e-8);

    dealii::SolverCG<> solver(solver_control);
//    Femenist::SolverSE<> solver(solver_control);
    
    solver .solve (
            this->system_equations.A,
            x,
            b
            ,dealii::PreconditionIdentity()
    );
    
    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

template<uint8_t dim>
prmt::Report HeatConductionProblemOnCell<dim>:: output_results ()
{
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (this->domain.dof_handler);

        char suffix[3] = {'x', 'y', 'z'};

        for (uint8_t i = 0; i < dim; ++i)
        {
            data_out.add_data_vector (/* heat_flow[i] */   solution[i], "solution");
            data_out.build_patches ();

            std::string file_name = output_file_name;
            file_name += suffix[i];//i;//boost::lexical_cast<char> (i);
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };
    };

    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (this->domain.dof_handler);

        char suffix[3] = {'x', 'y', 'z'};

        for (uint8_t i = 0; i < dim; ++i)
        {
            data_out.add_data_vector (heat_flow[i]  /* solution[i] */, "solution");
            data_out.build_patches ();

            std::string file_name = output_file_name;
            file_name += suffix[i];//i;//boost::lexical_cast<char> (i);
            file_name += "b.gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };
    };

    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler (this->domain.dof_handler);

        data_out.add_data_vector (anal_x, "solution");
        data_out.build_patches ();

        std::string file_name = "anal_x.gpd";

        std::ofstream output (file_name.data());
        data_out.write_gnuplot (output);
    };

    {
        char suffix[3] = {'x', 'y', 'z'};
        FOR_I(0, dim)
        {
            std::string file_name = "flow_";
            file_name += suffix[i];
            file_name += ".gpd";
            FILE *F;
            F = fopen(file_name.data(), "w");
            FOR_J(0, flow.size())
                fprintf(F, "%f %f %f %f\n", 
                        flow[j].first(0),
                        flow[j].first(1),
                        flow[j].second[i][0],
                        flow[j].second[i][1]);
            fclose(F);
        };
    };

    dbl meta[2][2] = {0};
    FOR_I(0, 2) FOR_J(0, 2) 
    {
        FOR_N(0, flow.size())
            meta[i][j] += (flow[n].second[i][j] * cells_area[n]);
        // meta[i][j] /= flow.size();
    };
        // FOR_N(0, flow.size())
        //     printf("%f\n", flow[n].second[1][1]);

    printf("META %f %f %f %f\n", 
            meta[0][0],
            meta[0][1],
            meta[1][0],
            meta[1][1]);

    REPORT_USE( 
            prmt::Report report;
            report.result = true;
            _return (report););
};

#endif









