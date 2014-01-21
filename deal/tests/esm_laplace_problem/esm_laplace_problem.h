/*
 * =====================================================================================
 *
 *       Filename:  esm_laplace_problem.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  21.09.2012 09:54:48
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef ELEMENT_STIFFNESS_MATRIX_LAPLACE_PROBLEM
#define ELEMENT_STIFFNESS_MATRIX_LAPLACE_PROBLEM

#include <projects/deal/main/element_stiffness_matrix/element_stiffness_matrix.desc.h>
#include <array>

template<uint8_t dim>
class HeatConductionProblemSup
{
    public:

        uint8_t static const num_coef = (dim * (dim + 1)) / 2;

        typedef std::array<std::vector<double>, num_coef > TypeCoef;
};

template<uint8_t dim>
class ElementStiffnessMatrixLaplaceProblem : 
    public ElementStiffnessMatrix< 
        dim, 
        double, 
        typename HeatConductionProblemSup<dim>::TypeCoef >
{
    public:
        ElementStiffnessMatrixLaplaceProblem ();

    virtual void set_coefficient (const typename HeatConductionProblemSup<dim>::TypeCoef &coef); 

    virtual double operator() (const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const;
};

template<uint8_t dim>
ElementStiffnessMatrixLaplaceProblem<dim>::
ElementStiffnessMatrixLaplaceProblem ()
    :
        ElementStiffnessMatrix<
        dim, 
        double, 
        typename HeatConductionProblemSup<dim>::TypeCoef > ()
{

};

template<uint8_t dim>
void ElementStiffnessMatrixLaplaceProblem<dim> :: 
set_coefficient (const  typename HeatConductionProblemSup<dim>::TypeCoef &coef)
{
    for (size_t i = 0; i < HeatConductionProblemSup<dim>::num_coef; ++i)
    {
        this->coefficient[i] .clear ();

        for (size_t j = 0; j < coef[i].size(); ++j)
            this->coefficient[i] .push_back (coef[i][j]);
    };
};

template<uint8_t dim>
double ElementStiffnessMatrixLaplaceProblem<dim> :: 
operator() (const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0.0;

    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j < dim; ++j)
                if (i == j)
                {
                    res += this->coefficient[i][material_id] * 
                        fe_values.shape_grad (index_i, q_point)[i] *
                        fe_values.shape_grad (index_j, q_point)[i] *
                        fe_values.JxW(q_point);
                }
                else
                {
                    uint8_t n = i + j + dim - 1;
                    res += this->coefficient[n][material_id] * 
                        fe_values.shape_grad (index_i, q_point)[i] *
                        fe_values.shape_grad (index_j, q_point)[j] *
                        fe_values.JxW(q_point); 
                };
//                    printf("%f %f %f\n", 
//                            fe_values.quadrature_point(0)(0),
//                            fe_values.quadrature_point(0)(1),
//                            fe_values.shape_grad(0,0)[0]);
    return res;
};

#endif
