/*
 * =====================================================================================
 *
 *       Filename:  solver_se.disc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11.09.2012 13:10:35
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef FEMENIST_SOLVER_SE_DESC

#define FEMENIST_SOLVER_SE_DESC

#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include </home/primat/deal.II/contrib/boost-1.49.0/include/boost/math/special_functions/sign.hpp>
#include </home/primat/deal.II/contrib/boost-1.49.0/include/boost/lexical_cast.hpp>

namespace Femenist
{

    template <class VECTOR = dealii::Vector<double> >
    class SolverSE
    {
        public:
            SolverSE (dealii::SolverControl &cn);

        //Method
        public:
            template <class MATRIX>
            prmt::Report solve (MATRIX &A,
                                VECTOR &x,
                                VECTOR &b);

        //Fields
        private:
            dealii::SolverControl solver_controller;

    };

};

#endif
