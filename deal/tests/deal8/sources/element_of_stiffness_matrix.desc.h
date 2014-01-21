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

class ElementStiffnessMatrix
{
    ElementStiffnessMatrix();

    double operator() (size_t index_i, size_t index_j, 
            QGauss<dim> &guadrature_formula, FEValues<dim> fe_values);
};

double ElementStiffnessMatrix::operator(size_t i, size_t j, 
            QGauss<dim> &guadrature_formula, FEValues<dim> fe_values)
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0;

    double coef[2];

    if (i == j) then
    {
        if (i & (size_t)0x1) then
        {
            coef[0] = coefficients(0,0);
            coef[1] = coefficients(1,1);
        }
        else
        {
            coef[0] = coefficients(3,3);
            coef[1] = coefficients(2,2);
        };
    }
    else
    {
        if (i & (size_t)0x1) then
        {
            coef[0] = coefficients(3,0);
            coef[1] = coefficients(2,1);
        }
        else
        {
            coef[0] = coefficients(0,3);
            coef[1] = coefficients(1,2);
        };
    };   

    if ((i * j) & (size_t)0x1) then
    for (uint8_t q_point = 0; q_point < num_quad_points; ++q_point)
    {
            res += 
                (coef[1] * 
                 (fe_values .shape_grad (i, q_point)[x] *
                  fe_values .shape_grad (j, q_point)[x]) +
                 coef[2] *
                 (fe_values .shape_grad (i, q_point)[y] *
                  fe_values .shape_grad (j, q_point)[y])
                ) * fe_values.JxW(q_point);
        else
            res += 
                (coef[1] * 
                 (fe_values .shape_grad (i, q_point)[x] *
                  fe_values .shape_grad (j, q_point)[y]) +
                 coef[2] *
                 (fe_values .shape_grad (i, q_point)[y] *
                  fe_values .shape_grad (j, q_point)[x])
                ) * fe_values.JxW(q_point);

    };
    
};
