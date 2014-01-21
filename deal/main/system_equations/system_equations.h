/*
 * =====================================================================================
 *
 *       Filename:  problem.h
 *
 *    Description:  :
 *
 *        Version:  1.0
 *        Created:  13.09.2012 10:27:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef SYSTEM_EQUATIONS_H

#define SYSTEM_EQUATIONS_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

template <class data_type>
struct SystemEquations
{
    void reinit (dealii::CompressedSparsityPattern &csp,
            size_t n_dofs)
    {
        this->sparsity_pattern .copy_from (csp);
        this->A .reinit (this->sparsity_pattern);
        this->x .reinit (n_dofs);
        this->b .reinit (n_dofs);
    };

    dealii::Vector          <data_type> x;
    dealii::Vector          <data_type> b;

    dealii::SparsityPattern sparsity_pattern;

    dealii::SparseMatrix    <data_type> A;
};
    struct OLOLO
    {
        dealii::SparseMatrix<double> sm;
        dealii::SparsityPattern      sp;
    };

#endif
