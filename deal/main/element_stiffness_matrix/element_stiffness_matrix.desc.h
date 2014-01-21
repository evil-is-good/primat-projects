/*
 * =====================================================================================
 *
 *       Filename:  element_of_stiffness_matrix.desc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04.09.2012 11:50:34
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef ELEMENT_STIFFNESS_MATRIX_H
#define ELEMENT_STIFFNESS_MATRIX_H

#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <projects/deal/main/function/function.h>

template<uint8_t dim, typename return_type, typename storage_coefficients_type>
class ElementStiffnessMatrix
{
    public:

    ElementStiffnessMatrix ();

    //Methods
    
    public:

    virtual void set_coefficient (const storage_coefficients_type &coef); 

//        virtual return_type operator() () const;
    virtual return_type operator() (
            const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const;

    //Fields

    protected:

    storage_coefficients_type coefficient;
};


template<uint8_t dim, class return_type, class storage_coefficients_type>
ElementStiffnessMatrix<dim, return_type, storage_coefficients_type>::
ElementStiffnessMatrix ()
{
};

template<uint8_t dim, class return_type, class storage_coefficients_type>
void ElementStiffnessMatrix<dim, return_type, storage_coefficients_type>::
set_coefficient (const storage_coefficients_type &coef)
{
    coefficient = coef;
};

template<uint8_t dim, class return_type, class storage_coefficients_type>
return_type ElementStiffnessMatrix<dim, return_type, 
            storage_coefficients_type>::
operator() (const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const
{
};







template<uint8_t dim, class return_type, class storage_coefficients_type>
class ElementRightHandSideVector
{
    public:

    ElementRightHandSideVector ();

    //Methods
    
    public:

    virtual void set_coefficient (const storage_coefficients_type &coef); 

//        virtual return_type operator() () const;
    virtual return_type operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_valuesi) const;

    //Fields

    protected:

    storage_coefficients_type coefficient;
};


template<uint8_t dim, class return_type, class storage_coefficients_type>
ElementRightHandSideVector<dim, return_type, storage_coefficients_type>::
 ElementRightHandSideVector()
{
};

template<uint8_t dim, class return_type, class storage_coefficients_type>
void ElementRightHandSideVector<dim, return_type, storage_coefficients_type>::
set_coefficient (const storage_coefficients_type &coef)
{
    coefficient = coef;
};

template<uint8_t dim, class return_type, class storage_coefficients_type>
return_type ElementRightHandSideVector<dim, return_type, 
            storage_coefficients_type>::
operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_valuesi) const
{
};

#endif
