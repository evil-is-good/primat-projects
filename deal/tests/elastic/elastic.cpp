/*
 * =====================================================================================
 *
 *       Filename:  elastic.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.09.2012 11:25:27
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include </home/primat/projects/deal/main/problem/problem.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

template <uint8_t dim>
class ElasticProblem
{
    public:
        ElasticProblem ();

    //Methods
    public:
        virtual Report solved ();


    protected:
        virtual Report setup_sysem ();
        virtual Report assemble_system ();
        virtual Report solve ();
        virtual Report output_results ();
};

template <uint8_t dim>
ElasticProblem<dim>::ElasticProblem ()
    :
        domain ()
        finite_element (FE_Q<dim>(1), dim)
{

};

template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
    domain .clear ();
};

template <int dim>
void ElasticProblem<dim>::setup_system ()
{
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());

    DoFTools:: imake_sparsity_pattern (dof_handler, c_sparsity);
    c_sparsity .compress ();
    sparsity_pattern .copy_from(c_sparsity);

    std::ofstream output ("matrix.out");
    sparsity_pattern.compress();
    sparsity_pattern .print_gnuplot (output);

    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
};


template <int dim>
void ElasticProblem<dim>::assemble_system ()
{
    QGauss<dim>  quadrature_formula(2);

    FEValues<dim> fe_values (fe, quadrature_formula,
            update_values   | update_gradients |
            update_quadrature_points | update_JxW_values);

    TestESM<dim> test_esm;
    double coef[2] = {1.0, 1.0};
    test_esm .set_coefficient(coef);



    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    std::vector<double>     lambda_values (n_q_points);
    std::vector<double>     mu_values (n_q_points);

    ConstantFunction<dim> lambda(1.), mu(1.);

    RightHandSide<dim>      right_hand_side;
    std::vector<Vector<double> > rhs_values (n_q_points,
            Vector<double>(dim));

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
             endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
        mu.value_list     (fe_values.get_quadrature_points(), mu_values);

        right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                rhs_values);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int
                component_i = fe.system_to_component_index(i).first;

            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int
                    component_j = fe.system_to_component_index(j).first;

                //              printf("i = %d, j = %d, res = %f\n", i, j,
                //                      test_esm (i, j, quadrature_formula, fe_values));

                cell_matrix(i,j) += test_esm (i, j,
                        quadrature_formula, fe_values);
                ////
                double res = 0;
                for (unsigned int q_point=0; q_point<n_q_points;
                        ++q_point)
                {
                    //		    cell_matrix(i,j)
                    res
                        +=
                        (
                         (fe_values.shape_grad(i,q_point)[component_i] *
                          fe_values.shape_grad(j,q_point)[component_j] *
                          lambda_values[q_point])
                         +
                         (fe_values.shape_grad(i,q_point)[component_j] *
                          fe_values.shape_grad(j,q_point)[component_i] *
                          mu_values[q_point])
                         +
                         ((component_i == component_j) ?
                          (fe_values.shape_grad(i,q_point) *
                           fe_values.shape_grad(j,q_point) *
                           mu_values[q_point])  :
                          0)
                        )
                        *
                        fe_values.JxW(q_point);
                    ////              printf("res=%f\n", res);

                }
            }
        }
        printf("!!!!!!!!!!!!!\n");

        // Assembling the right hand
        // side is also just as
        // discussed in the
        // introduction:
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int
                component_i = fe.system_to_component_index(i).first;

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                cell_rhs(i) += fe_values.shape_value(i,q_point) *
                    rhs_values[q_point](component_i) *
                    fe_values.JxW(q_point);
        }
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    //    hanging_node_constraints.condense (system_matrix);
    //    hanging_node_constraints.condense (system_rhs);
    for (size_t i = 0; i < system_matrix.m(); ++i)
        for (size_t j = 0; j < system_matrix.n(); ++j)
            if (system_matrix .el (i,j))
                printf("system_matrix_old(%d,%d)=%f\n", i,j,system_matrix(i,j));

    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
            0,
            ConstantFunction<dim>(0, dim),// 
            //ZeroFunction<dim>(dim),
            boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
            system_matrix,
            solution,
            system_rhs);
}
