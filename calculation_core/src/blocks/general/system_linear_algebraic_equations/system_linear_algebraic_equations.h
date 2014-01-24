#ifndef SYSTEM_LINEAR_ALGEBRAIC_EQUATION

#define SYSTEM_LINEAR_ALGEBRAIC_EQUATION

#include "../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

struct  SystemsLinearAlgebraicEquations 
{
    void matrix_reinit (dealii::CompressedSparsityPattern &csp)
    {
        this->sparsity_pattern .copy_from (csp);
        this->matrix .reinit (this->sparsity_pattern);
    };
    
    dealii::SparsityPattern   sparsity_pattern;
    dealii::SparseMatrix<dbl> matrix;
    dealii::Vector<dbl>       solution;
    dealii::Vector<dbl>       rhsv;
};

#endif
