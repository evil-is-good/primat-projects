/*
 * =====================================================================================
 *
 *       Filename:  problem.desc.h
 *
 *    Description:  :
 *
 *        Version:  1.0
 *        Created:  13.09.2012 10:30:22
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef PROBLEM_DESC_H

#define PROBLEM_DESC_H

#include <stdint.h>
#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <projects/deal/main/domain/domain.h>
#include <projects/deal/main/system_equations/system_equations.h>

template <uint8_t dim, class ELEMENT_S_MATRYX, class ELEMENT_RH_VECTOR>
class Problem
{
    public:
        Problem ();
        ~Problem ();

    //Methods
    public:
        virtual prmt::Report solved () = 0;

    protected:
//        virtual Report setup_sysem () = 0;
//        virtual Report assemble_system () = 0;
//        virtual Report apply_boundary_values () = 0;
//        virtual Report solve_system_equations () = 0;
//        virtual Report output_results () = 0;

    //Fields
    public:
        Domain<dim> domain;
        SystemEquations<double> system_equations;

    // protected:
    public:
        ELEMENT_S_MATRYX  element_stiffness_matrix;
        ELEMENT_RH_VECTOR element_rh_vector;
};

template<uint8_t dim, class ELEMENT_S_MATRYX, class ELEMENT_RH_VECTOR>
Problem<dim, ELEMENT_S_MATRYX, ELEMENT_RH_VECTOR>::Problem ()
    :
        domain ()
{

};

template<uint8_t dim, class ELEMENT_S_MATRYX, class ELEMENT_RH_VECTOR>
Problem<dim, ELEMENT_S_MATRYX, ELEMENT_RH_VECTOR>::~Problem ()
{

};


#endif
