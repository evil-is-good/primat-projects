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

#ifndef DOMAIN_H

#define DOMAIN_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>

template <uint8_t dim>
struct Domain
{
    Domain () : dof_handler (triangulation) {};

    void clean () {dof_handler .clear ();};

    dealii::Triangulation <dim> triangulation;
    dealii::DoFHandler    <dim> dof_handler;
};

#endif
