/*
 * =====================================================================================
 *
 *       Filename:  domain.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.09.2012 11:44:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef CALCULATION_CORE_DOMAIN
#define CALCULATION_CORE_DOMAIN

#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>

template <u8 dim>
struct Domain
{
    void dof_init () {dof_handler .initialize (grid, fe);};

    dealii::Triangulation <dim> grid;
    dealii::DoFHandler    <dim> dof_handler;
    dealii::FiniteElement <dim> fe;
};

#endif
