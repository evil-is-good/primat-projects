/*
 * =====================================================================================
 *
 *       Filename:  rhsv_on_cell.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12.10.2012 15:33:12
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <projects/deal/main/element_stiffness_matrix/element_stiffness_matrix.desc.h>

template<uint8_t dim>
class ElementRightHandSideVectorElasicProblemOnCell : 
    public ElementRightHandSideVector<
        dim, 
        double,
        std::array<std::array<std::vector<double>, dim>, dim> >
{
    public:
        ElementRightHandSideVectorElasicProblemOnCell ();

    virtual void set_coefficient (
            const std::array<std::array<std::vector<double>, dim>, dim> &coef);

    virtual double operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const;
};

template<uint8_t dim>
ElementRightHandSideVectorElasicProblemOnCell<dim>::
ElementRightHandSideVectorElasicProblemOnCell ()
    :
        ElementRightHandSideVector<
            dim, 
            double, 
            std::array<std::array<std::vector<double>, dim>, dim> > ()
{

};

template<uint8_t dim>
void ElementRightHandSideVectorElasicProblemOnCell<dim> :: 
set_coefficient (const std::array<std::array<std::vector<double>, dim>, dim> 
        &coef)
{
    for (size_t i = 0; i < coef.size(); ++i)
        for (size_t j = 0; j < coef[i].size(); ++j)
        { 
            this->coefficient[i][j] .clear ();

            for (size_t k = 0; k < coef[i][j].size(); ++k)
                this->coefficient[i][j] .push_back (coef[i][j][k]);
        };
};

template<uint8_t dim>
double ElementRightHandSideVectorElasicProblemOnCell<dim> :: 
operator() (const size_t index_i, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0.0;

    size_t a = index_i % dim;

//    printf("Qadrature: index=%ld\n", index_i);
    for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
    {
        for(size_t i = 0; i < this->coefficient.size(); ++i)
        {
            if (fabs(this->coefficient[a][i][material_id]) > 1e-12)
                res += -fe_values.shape_grad (index_i, q_point)[i] *
                    this->coefficient[a][i][material_id] *
                    fe_values.JxW(q_point);

//            printf("coor_%ld=%f,",fe_values.quadrature_point(q_point)[i]);
//            printf(" coef = %f,", 
//                    this->coefficient[a][i][material_id]);
//            printf(" grad = %f\n", fe_values.shape_grad (index_i,q_point )[i]);
        };
    };
//    printf("\n");
//    printf("%ld %f\n", index_i, res);

    return res;
};
