/*
 * =====================================================================================
 *
 *       Filename:  elastic_problem_2d_on_cell.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  24.10.2012 15:21:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef ELASTIC_PROBLEM_2D_ON_CELL

#define ELASTIC_PROBLEM_2D_ON_CELL

#include <projects/deal/tests/elastic_problem_plane_deformation_on_cell/elastic_problem_plane_deformation_on_cell.h>
#include <projects/deal/tests/heat_conduction_problem_on_cell/heat_conduction_problem_on_cell.h>

const uint8_t dim = 2; 

class ElasticProblem2DOnCell 
{
    private:
        static const uint8_t x = 0;
        static const uint8_t y = 1;
        static const uint8_t z = 2;

    public:
        ElasticProblem2DOnCell (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim+1>::TypeCoef &coef);
        
    //Methods
    public:
        Report solved ();
        void print_result (
                const std::string &filename_from_problem_1,
                const std::string &filename_from_problem_2);

    private:
        typename ElasticProblemSup<dim>::TypeCoef coef_to_problem_1_coef 
            (const typename ElasticProblemSup<dim+1>::TypeCoef &coef) const;
        typename HeatConductionProblemSup<dim>::TypeCoef coef_to_problem_2_coef 
            (const typename ElasticProblemSup<dim+1>::TypeCoef &coef) const;

    //Fields
    public:
        double meta_coefficient[dim+1][dim+1][dim+1][dim+1];
        
    private:
        typename ElasticProblemSup<dim+1>::TypeCoef coefficient;
        
        ElasticProblemPlaneDeformationOnCell<dim> problem_1;
        HeatConductionProblemOnCell<dim> problem_2;
};

ElasticProblem2DOnCell::ElasticProblem2DOnCell (
                const dealii::Triangulation<dim> &triangulation,
                const typename ElasticProblemSup<dim+1>::TypeCoef &coef)
:
    problem_1 (triangulation, coef_to_problem_1_coef (coef)),
    problem_2 (triangulation, coef_to_problem_2_coef (coef))
{
    for (size_t i = 0; i < dim + 1; ++i)
        for (size_t j = 0; j < dim + 1; ++j)
            for (size_t k = 0; k < dim + 1; ++k)
                for (size_t l = 0; l < dim + 1; ++l)
                {
                    this->coefficient[i][j][k][l] .clear ();

                    for (size_t m = 0; m < coef[i][j][k][l].size(); ++m)
                        this->coefficient[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
                };
};

typename ElasticProblemSup<dim>::TypeCoef 
ElasticProblem2DOnCell::coef_to_problem_1_coef (
        const typename ElasticProblemSup<dim+1>::TypeCoef &coef) const
{
    ElasticProblemSup<dim>::TypeCoef coef_for_problem_1;

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                {
                    coef_for_problem_1[i][j][k][l] .clear ();

                    for (size_t m = 0; m < coef[i][j][k][l].size(); ++m)
                        coef_for_problem_1[i][j][k][l] 
                            .push_back (coef[i][j][k][l][m]);
                };

    return coef_for_problem_1;
};

typename HeatConductionProblemSup<dim>::TypeCoef
ElasticProblem2DOnCell::coef_to_problem_2_coef (
        const typename ElasticProblemSup<dim+1>::TypeCoef &coef) const
{
    HeatConductionProblemSup<dim>::TypeCoef coef_for_problem_2;

    const uint8_t xx = 0;
    const uint8_t yy = 1;
    const uint8_t xy = 2;

    coef_for_problem_2[xx] .clear ();
    coef_for_problem_2[yy] .clear ();
    coef_for_problem_2[xy] .clear ();

    for (size_t i = 0; i < coef[0][0][0][0].size(); ++i)
    {
        coef_for_problem_2[xx] .push_back (coef[x][z][x][z][i]);
        coef_for_problem_2[yy] .push_back (coef[y][z][y][z][i]);
        coef_for_problem_2[xy] .push_back (coef[x][z][y][z][i]);
    };

    return coef_for_problem_2;
};

Report ElasticProblem2DOnCell::solved ()
{
    problem_1 .solved ();

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            for (size_t k = 0; k < dim; ++k)
                for (size_t l = 0; l < dim; ++l)
                    meta_coefficient[i][j][k][l] = 
                        problem_1 .meta_coefficient[i][j][k][l];

    std::array<std::array<std::vector<double>, dim>, dim> coef_for_rhs;
    for(size_t i = 0; i < dim; ++i)
        for(size_t j = 0; j < dim; ++j)
            coef_for_rhs[i][j] .resize (coefficient[i][j][0][0].size());

    for(size_t i = 0; i < dim; ++i)
        for(size_t j = 0; j < dim; ++j)
            for(size_t k = 0; 
                    k < coefficient[i][j][z][z].size(); ++k)
            {
                coef_for_rhs[i][j][k] = 
                    coefficient[i][j][z][z][k];
            };

    problem_1.system_equations.b = 0;

    problem_1.element_rh_vector .set_coefficient (coef_for_rhs);

    REPORT problem_1 .assemble_right_vector_of_system ();

//    REPORT problem_1 .solve_system_equations ();
//
//    for (size_t i = 0; i < problem_1.system_equations.x.size(); ++i)
//        problem_1.system_equations.x(i) = problem_1.system_equations.x(
//                problem_1.black_on_white_substituter .subst (i));

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
        {
            for (size_t k = 0; k < coefficient[i][j][z][z].size(); ++k)
                meta_coefficient[i][j][z][z] +=
                    coefficient[i][j][z][z][k] * 
                    problem_1.area_of_material[k];

            meta_coefficient[i][j][z][z] /= problem_1.area_of_domain;
        };

    {
        size_t len_vector_solution = problem_1.domain.dof_handler.n_dofs();
        uint8_t width_2d_matrix = dim * dim;
        double mean_stress[dim][dim];

        for (size_t i = 0; i < width_2d_matrix; ++i)
        {
            uint8_t im = i % dim;
            uint8_t in = i / dim;

            mean_stress[im][in] = 0.0;

            uint8_t a = im;
            uint8_t b = in;

            if (im == 1)
                a ^= b ^= a ^= b;

            for (size_t k = 0; k < len_vector_solution; ++k)
                mean_stress[im][in] += 
                    problem_1.solution[a][b](k) * (-problem_1.system_equations.b(k));

            //            printf("STRESS=%f\n", mean_stress[im][in][jm][jn]);

            mean_stress[im][in] /= problem_1.area_of_domain; 

            meta_coefficient[im][in][z][z] += 
                mean_stress[im][in];

            meta_coefficient[z][z][im][in] = 
                meta_coefficient[im][in][z][z];
        };
    };

    problem_2 .solved ();

    const uint8_t xx = 0;
    const uint8_t yy = 1;
    const uint8_t xy = 2;

    meta_coefficient[x][z][x][z] = problem_2 .meta_coefficient[xx];
    meta_coefficient[y][z][y][z] = problem_2 .meta_coefficient[yy];
    meta_coefficient[x][z][y][z] = problem_2 .meta_coefficient[xy];

    for (size_t k = 0; k < coefficient[z][z][z][z].size(); ++k)
        meta_coefficient[z][z][z][z] +=
            coefficient[z][z][z][z][k] * 
            problem_1.area_of_material[k];

    meta_coefficient[z][z][z][z] /= problem_1.area_of_domain;

};

void ElasticProblem2DOnCell::print_result (
        const std::string &filename_from_problem_1,
        const std::string &filename_from_problem_2)
{
    problem_1 .print_result (filename_from_problem_1);
    problem_2 .print_result (filename_from_problem_2);
};

#endif
