#ifndef TRIVIAL_PREPARE_SYSTEM_EQUATIONS
#define TRIVIAL_PREPARE_SYSTEM_EQUATIONS

#include "../../domain/domain.h"
#include "../../system_linear_algebraic_equations/system_linear_algebraic_equations.h"

//! Дополнительный инструментарий
/*!
 * Additional tools
 */
namespace ATools
{
    template<u8 dim>
    void trivial_prepare_system_equations (
            SystemsLinearAlgebraicEquations<dim> &se,
            const Domain<dim> &domain)
    {
        dealii::CompressedSparsityPattern c_sparsity (
                domain.dof_handler.n_dofs());

        dealii::DoFTools ::make_sparsity_pattern (
                domain.dof_handler, c_sparsity);

        se .matrix_reinit (c_sparsity);

        se.solution .reinit (domain.dof_handler .n_dofs());
        se.rhsv     .reinit (domain.dof_handler .n_dofs());
    };
};

#endif
