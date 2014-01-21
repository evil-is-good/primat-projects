//!  A test class. 
/*!
  A more elaborate class description.
*/
#ifndef ELEMENT_STIFFNESS_MATRIX_ELASTIC_PROBLEM
#define ELEMENT_STIFFNESS_MATRIX_ELASTIC_PROBLEM

#include <stdint.h>

#include <vector>
#include <array>
#include <projects/deal/main/element_stiffness_matrix/element_stiffness_matrix.desc.h>

template<uint8_t dim>
class ElasticProblemSup
{
    public:
        typedef std::array<
            std::array<
            std::array<
            std::array<
            std::vector<double>
            ,dim>,dim>,dim>,dim> TypeCoef;

        typedef Femenist::Function<std::array<double, dim>, dim> TypeFunc;

        class MyFuncFromDealii : public dealii::Function<dim>
        {
            public:
                typedef std::function<std::array<double, dim>
                    (const dealii::Point<dim>&)> Func;

            public:
                MyFuncFromDealii () : dealii::Function<dim>() {};

                MyFuncFromDealii (Func func)
                    : dealii::Function<dim>() 
                {
                    this->func = func;
                };

                MyFuncFromDealii (const MyFuncFromDealii& mffd)
                {
                    this->func = mffd.func;
                };

                void operator= (Func func)
                {
                    this->func = func;
                };

                virtual void vector_value (const dealii::Point<dim> &p,
                                           dealii::Vector<double>   &values) const
                {
                    std::array<double, dim> res = this->func(p);

                    for (size_t i = 0; i < dim; ++i)
                        values(i) = res[i];
                };

            private:
                Func func;
        };

        struct BoundaryValues
        {
            MyFuncFromDealii function;
            size_t boundari_indicator;
            uint8_t type;

            static const uint8_t Dirichlet = 0;
            static const uint8_t Neumann   = 1;
        };
};

//!  A test class. 
/*!
  A more elaborate class description.
  \f$(x_1,y_1)\f$
  \f[
  \sum_{j,k=0}^2 C_{bjkd}\frac{\partial\phi_{ab}}{\partial j}
  \frac{\phi_{cd}}{\partial k}
  \f]
*/
template<uint8_t dim>
class ElementStiffnessMatrixElasticProblem : 
    public ElementStiffnessMatrix< 
        dim, 
        double, 
        typename ElasticProblemSup<dim>::TypeCoef>
{
    public:
        ElementStiffnessMatrixElasticProblem ();

    virtual void set_coefficient (
            const typename ElasticProblemSup<dim>::TypeCoef
            &coef); 

    virtual double operator() (const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const;
};

template<uint8_t dim>
ElementStiffnessMatrixElasticProblem<dim>::
    ElementStiffnessMatrixElasticProblem ()
    :
        ElementStiffnessMatrix<
        dim, 
        double, 
        typename ElasticProblemSup<dim>::TypeCoef > ()
{

};

template<uint8_t dim>
void ElementStiffnessMatrixElasticProblem<dim> :: 
set_coefficient (const typename ElasticProblemSup<dim>::TypeCoef &coef)
{
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                {
                    this->coefficient[i][j][k][l] .clear ();

                    for (size_t m = 0; m < coef[i].size(); ++m)
                        this->coefficient[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
                };
};

template<uint8_t dim>
double ElementStiffnessMatrixElasticProblem<dim> :: operator() (
            const size_t index_i, const size_t index_j, 
            const dealii::QGauss<dim> &quadrature_formula, 
            const dealii::FEValues<dim> &fe_values,
            const size_t material_id) const
{
    const uint8_t num_quad_points = quadrature_formula.size();

    double res = 0.0;

    size_t a = index_i % dim;
    size_t b = index_j % dim;

    for (uint8_t q_point = 0; q_point < num_quad_points; ++q_point)
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j < dim; ++j)
            {
                if (this->coefficient[a][i][j][b][material_id] > 1e-12)
                    res += 
                        (this->coefficient[a][i][j][b][material_id] * 
                         (fe_values .shape_grad (index_i, q_point)[i] *
                          fe_values .shape_grad (index_j, q_point)[j])
                        ) * fe_values.JxW(q_point);

                if (this->coefficient[b][i][j][a][material_id] > 1e-12)
                    res += 
                        (this->coefficient[b][i][j][a][material_id] *
                         (fe_values .shape_grad (index_j, q_point)[i] *
                          fe_values .shape_grad (index_i, q_point)[j])
                        ) * fe_values.JxW(q_point);
            };

    res /= 2.0;

//    printf("i=%d j=%d cond=%d a=%f b=%f res=%f\n", 
//            component_i, component_j, cond, a, b, res);

    return res;

};

#endif
