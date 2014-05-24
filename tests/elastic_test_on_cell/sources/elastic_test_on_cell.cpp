//#include "cgal/cgal.h"
//#include <stdlib.h>
//
////#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
////#include <CGAL/Constrained_Delaunay_triangulation_2.h>
////#include <CGAL/Delaunay_mesher_2.h>
////#include <CGAL/Delaunay_mesh_face_base_2.h>
////#include <CGAL/Delaunay_mesh_size_criteria_2.h>
//
////#include <projects/deal/tests/elastic_problem_plane_deformation_on_cell/elastic_problem_plane_deformation_on_cell.h>
//#include <projects/deal/tests/elastic_problem_2d_on_cell_v2/elastic_problem_2d_on_cell.h>
//#include <deal.II/grid/grid_generator.h>
//#include <deal.II/grid/tria_accessor.h>
//#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/grid_out.h>
//#include <deal.II/grid/grid_reordering.h>
//#include <ctime>
//
//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Triangulation_vertex_base_2<K> Vb;
//typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
//typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
//typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
//typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
//
//typedef CDT::Vertex_handle Vertex_handle;
//typedef CDT::Point Point;

#include "./head.h"

size_t name_main_stress = 0;

double glob_i = 0.0;

template <uint8_t dim>
double get_minor(const std::array<std::array<double, dim>, dim> &matrix,
        const size_t i_index, const size_t j_index);

template <uint8_t dim>
double get_determinant(const std::array<std::array<double, dim>, dim> &matrix);

//void foo_1 (
//        const dealii::Triangulation<dim> &triangulation)
//{
//    const uint8_t dim = 2;
//
//    ElasticProblemSup<dim>::TypeCoef coef;
//
//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = 0; j < dim; ++j)
//            for (size_t k = 0; k < dim; ++k)
//                for (size_t l = 0; l < dim; ++l)
//                    coef[i][j][k][l] .resize (2);
//
//    double yung;
//    double puasson;
//
//    double lambda;
//    double mu    ;
//
//    yung = 2.0;
//    puasson = 0.25;
//
//    lambda = 1.0;//(puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
//    mu     = 1.0;//yung / (2 * (1 + puasson));
//
//    coef[0][0][0][0][0] = 1.0;//lambda + 2 * mu;
//    coef[1][1][1][1][0] = 1.0;//lambda + 2 * mu;
//
//    coef[0][0][1][1][0] = lambda;
//    coef[1][1][0][0][0] = lambda;
//
//    coef[0][1][0][1][0] = mu;
//    coef[1][0][1][0][0] = mu;
//    coef[0][1][1][0][0] = mu;
//    coef[1][0][0][1][0] = mu;
//
//    yung = 2.0;
//    puasson = 0.25;
//
//    lambda = 1.0;//(puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
//    mu     = 1.0;//yung / (2 * (1 + puasson));
//
//    coef[0][0][0][0][1] = 2.0;//(lambda + 2 * mu) * 2;
//    coef[1][1][1][1][1] = 1.0;//lambda + 2 * mu;
//
//    coef[0][0][1][1][1] = lambda;
//    coef[1][1][0][0][1] = lambda;
//
//    coef[0][1][0][1][1] = mu;
//    coef[1][0][1][0][1] = mu;
//    coef[0][1][1][0][1] = mu;
//    coef[1][0][0][1][1] = mu;
//    
//    ElasticProblemPlaneDeformationOnCell<dim> problem (
//            triangulation, coef);
//
//    REPORT problem .solved ();
//    
//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = 0; j < dim; ++j)
//            for (size_t k = 0; k < dim; ++k)
//                for (size_t l = 0; l < dim; ++l)
//                    printf("coef[%ld][%ld][%ld][%ld]=%f\n", i,j,k,l,
//                            coef[i][j][k][l][0]);
//
//    printf("\n");
//
//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = 0; j < dim; ++j)
//            for (size_t k = 0; k < dim; ++k)
//                for (size_t l = 0; l < dim; ++l)
//                    printf("mean[%ld][%ld][%ld][%ld]=%f\n", i,j,k,l,
//                            problem.mean_coefficient[i][j][k][l]);
//
//    printf("\n");
//
//    for (size_t i = 0; i < dim; ++i)
//        for (size_t j = 0; j < dim; ++j)
//            for (size_t k = 0; k < dim; ++k)
//                for (size_t l = 0; l < dim; ++l)
//                    printf("meta[%ld][%ld][%ld][%ld]=%f\n", i,j,k,l,
//                            problem.meta_coefficient[i][j][k][l]);
//
//    problem .print_result ("output-");
//
//};

//typename ElasticProblemSup<3>::TypeCoef foo (
//        typename ElasticProblemSup<3>::TypeCoef a)
//{
//    typename ElasticProblemSup<3>::TypeCoef res;
//
//    double A = 
//        1 - 
//        (unphys[x][y] * unphys[y][x] + unphys[z][x] * unphys[x][z] + unphys[y][z] * unphys[z][y]) -
//        (unphys[x][y] * unphys[y][z] * unphys[z][x] + unphys[x][z] * unphys[z][y] * unphys[y][x]);
//
//    for (uint8_t i = 0; i < 3; ++i)
//    {
//        int no_1 = (i + 1) % 3;
//        int no_2 = (i + 2) % 3;
//
//        for (uint8_t j = 0; j < 3; ++j)
//        {
//            if (i == j)
//                res.c[i][j] = (1 - unphys[no_1][no_2] * unphys[no_2][no_1]);
//            else
//            {
//                int k = (j == no_1) ? no_2 : no_1;
//                res.c[i][j] = (unphys[j][i] + unphys[j][k] * unphys[k][i]);
//            };
//
//            res.c[i][j] *= (unphys[i][i] / A);
//        };
//    };
//        
//    return res;
//};

class Hexagon
{
    public:
        Hexagon(): center(0.0, 0.0), radius(0.0) {};
        Hexagon(const dealii::Point<2> &c, const double r)
        {
            center = c;
            radius = r;

            point[0] = dealii::Point<2>(center(0), center(1) - radius);
            point[1] = dealii::Point<2>(center(0) + radius * (sqrt(3.0) / 2.0),
                                             center(1) - radius / 2.0);
            point[2] = dealii::Point<2>(center(0) + radius * (sqrt(3.0) / 2.0), 
                                             center(1) + radius / 2.0);
            point[3] = dealii::Point<2>(center(0), center(1) + radius);
            point[4] = dealii::Point<2>(center(0) - radius * (sqrt(3.0) / 2.0), 
                                             center(1) + radius / 2.0);
            point[5] = dealii::Point<2>(center(0) - radius * (sqrt(3.0) / 2.0),
                                             center(1) - radius / 2.0);
        };

        bool include (const dealii::Point<2> &p)
        {
            bool res = false;

            if (radius < 1e-10)
                return res;

//            printf("%f %f %f %f %f %f\n", p1(0), p2(0), p3(0), p4(0), p5(0), p6(0));
//            printf("%f %f %f %f %f %f\n", p1(1), p2(1), p3(1), p4(1), p5(1), p6(1));

            auto above_the_line = 
                [p] (const dealii::Point<2, double> pl, 
                        const dealii::Point<2, double> pr) -> bool
                {
                    const uint8_t x = 0;
                    const uint8_t y = 1;

                    if ((
                                pl(x) * pr(y) - 
                                pr(x) * pl(y) + 
                                p(x)  * (pl(y) - pr(y)) + 
                                p(y)  * (pr(x) - pl(x))) >= -1e-12)
                        return true;
                    else
                        return false;
                };

            size_t sum_positive_sign = 
                above_the_line(point[0], point[1]) +
                above_the_line(point[1], point[2]) +
                above_the_line(point[2], point[3]) +
                above_the_line(point[3], point[4]) +
                above_the_line(point[4], point[5]) +
                above_the_line(point[5], point[0]);

//            printf("%f %f %f %f %f %f\n", 
//                above_the_line(p1, p2),
//                above_the_line(p2, p3),
//                above_the_line(p3, p4),
//                above_the_line(p5, p6),
//                above_the_line(p6, p1));

            if (sum_positive_sign == 6)
                res = true;

            return res;
        };

    std::array<dealii::Point<2>, 6> point;

    private:
        dealii::Point<2> center;
        double radius;
};

typename ElasticProblemSup<3>::TypeCoef matrix_multiplication(
        typename ElasticProblemSup<3>::TypeCoef &a,
        typename ElasticProblemSup<3>::TypeCoef &b)
{
    typename ElasticProblemSup<3>::TypeCoef res;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    res[i][j][k][l] .resize (1);

    std::array<std::array<double, 9>, 9> A;
    std::array<std::array<double, 9>, 9> B;
    std::array<std::array<double, 9>, 9> C;

    uint8_t width_2d_matrix = 9;

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / 3;
        uint8_t in = i % 3;

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / 3;
            uint8_t jn = j % 3;

            A[i][j] = a[im][in][jm][jn][0];
        };
    };

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / 3;
        uint8_t in = i % 3;

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / 3;
            uint8_t jn = j % 3;

            B[i][j] = b[im][in][jm][jn][0];
        };
    };

    FOR_I(0, width_2d_matrix)
        FOR_J(0, width_2d_matrix)
        {
            double temp = 0.0;
            FOR_K(0, width_2d_matrix)
                temp += A[i][k] * B[k][j];
            C[i][j] = temp;
        };

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / 3;
        uint8_t in = i % 3;

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / 3;
            uint8_t jn = j % 3;

            res[im][in][jm][jn][0] = C[i][j];
        };
    };

    return res;
};

template<size_t size>
std::array<size_t, 2> to2D(const size_t i)//, const size_t j)
{
    std::array<size_t, 2> res;

    switch (size)
    {
        case 6*6:
            {
                if (i < 3)
                {
                    res[0] = i;
                    res[1] = i;
                }
                else
                {
                    switch (i)
                    {
                        case 3: res[0]=1; res[1]=2; break; 
                        case 4: res[0]=2; res[1]=0; break; 
                        case 5: res[0]=0; res[1]=1; break;
                    };
                };
            };
            break;
        case 9*9:
            {
                    res[0] = i / 3;
                    res[1] = i % 3;
            };
            break;
    };

    return res;

//    if (i == j)
//        return i;
//    if ((i + j) == 3)
//        return 4;
//    if ((i + j) == 2)
//        return 5;
//    if ((i + j) == 1)
//        return 6;
}

bool check_tetragonality(
        typename ElasticProblemSup<3>::TypeCoef &c,
        typename ElasticProblemSup<3>::TypeCoef &s)
{
    std::array<std::array<double, 6>, 6> C;
    std::array<std::array<double, 6>, 6> S;

    uint8_t width_2d_matrix = 6;

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        auto ind = to2D<6*6>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            auto jnd = to2D<6*6>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            C[i][j] = c[im][in][jm][jn][0];
            S[i][j] = s[im][in][jm][jn][0];
        };
    };

    double sd = S[2][2] * (S[0][0] + S[0][1]) - 2.0 * pow(S[0][2], 2.0);

    printf("%f %f %f %f %f %f\n", 
            (C[0][0] + C[0][1]) - (S[2][2] / sd),
            (C[0][0] - C[0][1]) - (1.0 / (S[0][0] - S[0][1])),
            (C[0][2]) - (-S[0][2] / sd),
            (C[2][2]) - ((S[0][0] + S[0][1]) / sd),
            (C[3][3]) - (1.0 / S[3][3]),
            (C[5][5]) - (1.0 / S[5][5]));

    if (
            (C[0][0] + C[0][1] == S[2][2] / sd) &&
            (C[0][0] - C[0][1] == 1.0 / (S[0][0] - S[0][1])) &&
            (C[0][2] == -S[0][2] / sd) &&
            (C[2][2] == (S[0][0] - S[0][1]) / sd) &&
            (C[3][3] == 1.0 / S[3][3]) &&
            (C[5][5] == 1.0 / S[5][5])
        ) 
        return true;
    else
        return false;
};

void turn(
        typename ElasticProblemSup<3>::TypeCoef &c,
        const double angle)
{
    double t[3][3][3][3] = {0.0};

    double a[3][3] = {
        {cos(angle), -sin(angle), 0},
        {sin(angle),  cos(angle), 0},
        {0,           0,          1}};

    FOR_I(0, 3) FOR_J(0, 3) FOR_K(0, 3) FOR_L(0, 3)
    {
        t[i][j][k][l] = 0.0;

        FOR_M(0, 3) FOR_N(0, 3) FOR_O(0, 3) FOR_P(0, 3)
        {
            t[i][j][k][l] +=  a[i][m] * a[j][n] * a[k][o] * a[l][p] * c[m][n][o][p][0];
//            if ((i == 0) &&
//                (j == 0) &&
//                (k == 0) &&
//                (l == 1) )
//                printf("%ld %ld %ld %ld %f %f\n", m,n,o,p,a[i][m] * a[j][n] * a[k][o] * a[l][p] *c[m][n][o][p][0], c[m][n][o][p][0]);
        };
        const double pi = 3.14159265359;
//            if ((i == 0) &&
//                (j == 0) &&
//                (k == 0) &&
//                (l == 1) )
//                printf("%f %f\n", 
//a[i][1] * a[j][1] * a[k][0] * a[l][0],// + a[i][1] * a[j][1] * a[k][1] * a[l][1],
//                        (pow(cos(pi/8.0), 3) * sin(pi/8.0) - pow(sin(pi/8.0), 3) * cos(pi/8.0)));
    };

    FOR_I(0, 3) FOR_J(0, 3) FOR_K(0, 3) FOR_L(0, 3)
        c[i][j][k][l][0] = t[i][j][k][l];
};

bool check_hexagonality(
        typename ElasticProblemSup<3>::TypeCoef &c,
        typename ElasticProblemSup<3>::TypeCoef &s)
{
    std::array<std::array<double, 6>, 6> C;
    std::array<std::array<double, 6>, 6> S;

    uint8_t width_2d_matrix = 6;

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        auto ind = to2D<6*6>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            auto jnd = to2D<6*6>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            C[i][j] = c[im][in][jm][jn][0];
            S[i][j] = s[im][in][jm][jn][0];
        };
    };

    double sd = S[2][2] * (S[0][0] + S[0][1]) - 2.0 * pow(S[0][2], 2.0);

    printf("%f %f %f %f %f %f\n", 
            (C[0][0] + C[0][1]) - (S[2][2] / sd),
            (C[0][0] - C[0][1]) - (1.0 / (S[0][0] - S[0][1])),
            (C[0][2]) - (-S[0][2] / sd),
            (C[2][2]) - ((S[0][0] - S[0][1]) / sd),
            (C[3][3]) - (1.0 / S[3][3]),
            (C[5][5]) - (1.0 / S[5][5]));

    if (
            (C[0][0] + C[0][1] == S[2][2] / sd) &&
            (C[0][0] - C[0][1] == 1.0 / (S[0][0] - S[0][1])) &&
            (C[0][2] == -S[0][2] / sd) &&
            (C[2][2] == (S[0][0] - S[0][1]) / sd) &&
            (C[3][3] == 1.0 / S[3][3]) &&
            (C[5][5] == 1.0 / S[5][5])
        ) 
        return true;
    else
        return false;
};

template<size_t size>
typename ElasticProblemSup<3>::TypeCoef unphysical_to_compliance(
        typename ElasticProblemSup<3>::TypeCoef unphys)
{
    const size_t width = static_cast<size_t>(sqrt(size));

    typename ElasticProblemSup<3>::TypeCoef res;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    res[i][j][k][l] .resize (1);

    std::array<std::array<double, width>, width> elastic_matrix;
    std::array<std::array<double, width>, width> compliance_matrix;

    for (size_t i = 0; i < width; ++i)
    {
        auto ind = to2D<size>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width; ++j)
        {
            auto jnd = to2D<size>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            elastic_matrix[i][j] = unphys[im][in][jm][jn][0];
        };
    };

    double det = get_determinant<width>(elastic_matrix);

    printf("DDDDDDD=%f\n", det);

    FOR_I(0, width)
        FOR_J(0, width)
            compliance_matrix[i][j] = 
                pow(-1, i + j) * get_minor<width>(elastic_matrix, j, i) / det;

    for (size_t i = 0; i < width; ++i)
    {
        auto ind = to2D<size>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width; ++j)
        {
            auto jnd = to2D<size>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            res[im][in][jm][jn][0] = compliance_matrix[i][j];
        };
    };

    return res;
};

void set_coef (
        typename ElasticProblemSup<3>::TypeCoef &coef,
        const uint8_t id,
        const double yung, const double puasson)
{
    const double lambda = 
        (puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
    const double mu     = 
        yung / (2 * (1 + puasson));

    coef[x][x][x][x][id] = lambda + 2 * mu;
    coef[y][y][y][y][id] = lambda + 2 * mu;
    coef[z][z][z][z][id] = lambda + 2 * mu;

    coef[x][x][y][y][id] = lambda;
    coef[y][y][x][x][id] = lambda;

    coef[x][y][x][y][id] = mu;
    coef[y][x][y][x][id] = mu;
    coef[x][y][y][x][id] = mu;
    coef[y][x][x][y][id] = mu;

    coef[x][x][z][z][id] = lambda;
    coef[z][z][x][x][id] = lambda;

    coef[x][z][x][z][id] = mu;
    coef[z][x][z][x][id] = mu;
    coef[x][z][z][x][id] = mu;
    coef[z][x][x][z][id] = mu;

    coef[z][z][y][y][id] = lambda;
    coef[y][y][z][z][id] = lambda;

    coef[z][y][z][y][id] = mu;
    coef[y][z][y][z][id] = mu;
    coef[z][y][y][z][id] = mu;
    coef[y][z][z][y][id] = mu;
};

template <uint8_t dim>
double get_minor(const std::array<std::array<double, dim>, dim> &matrix,
        const size_t i_index, const size_t j_index)
{
    std::array<std::array<double, dim-1>, dim-1> less_matrix;

    size_t i_count = 0;
    FOR_I(0, dim)
    {
        size_t j_count = 0;
        if (i != i_index)
        {
            FOR_J(0, dim)
            {
                if (j != j_index)
                {
                    less_matrix[i_count][j_count] = matrix[i][j];
                    j_count++;
                };
            };
            i_count++;
        };
    };

    return get_determinant<dim-1>(less_matrix);
};

template <>
double get_minor<2>(const std::array<std::array<double, 2>, 2> &matrix,
        const size_t i_index, const size_t j_index)
{
    return matrix[!i_index][!j_index];
};

template <uint8_t dim>
double get_determinant(const std::array<std::array<double, dim>, dim> &matrix)
{
    double res = 0.0;

    FOR_I(0, dim)
        res += pow(-1.0, i) * matrix[0][i] * get_minor<dim> (matrix, 0, i);

    return res;
};

template <>
double get_determinant<1>(const std::array<std::array<double, 1>, 1> &matrix)
{
    return matrix[0][0];
};

template <uint8_t dim>
void foo_2 (
        const dealii::Triangulation<dim> &triangulation)
{
    const uint8_t x = 0;
    const uint8_t y = 1;
    const uint8_t z = 2;

    typename ElasticProblemSup<dim + 1>::TypeCoef coef;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                    coef[i][j][k][l] .resize (2);

//    double yung;
//    double puasson;
//
//    double lambda;
//    double mu    ;
//
//    yung = 1.0;
//    puasson = 0.25;
//
//    lambda = (puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
//    mu     = yung / (2 * (1 + puasson));
//
//    coef[x][x][x][x][0] = lambda + 2 * mu;
//    coef[y][y][y][y][0] = lambda + 2 * mu;
//    coef[z][z][z][z][0] = lambda + 2 * mu;
//
//    coef[x][x][y][y][0] = lambda;
//    coef[y][y][x][x][0] = lambda;
//
//    coef[x][y][x][y][0] = mu;
//    coef[y][x][y][x][0] = mu;
//    coef[x][y][y][x][0] = mu;
//    coef[y][x][x][y][0] = mu;
//
//    coef[x][x][z][z][0] = lambda;
//    coef[z][z][x][x][0] = lambda;
//
//    coef[x][z][x][z][0] = mu;
//    coef[z][x][z][x][0] = mu;
//    coef[x][z][z][x][0] = mu;
//    coef[z][x][x][z][0] = mu;
//
//    coef[z][z][y][y][0] = lambda;
//    coef[y][y][z][z][0] = lambda;
//
//    coef[z][y][z][y][0] = mu;
//    coef[y][z][y][z][0] = mu;
//    coef[z][y][y][z][0] = mu;
//    coef[y][z][z][y][0] = mu;
//
//    yung = 2.0;
//    puasson = 0.25;
//
//    lambda = (puasson * yung) / ((1 + puasson) * (1 - 2 * puasson));
//    mu     = yung / (2 * (1 + puasson));
//
//    coef[x][x][x][x][1] = (lambda + 2 * mu);// * 2;
//    coef[y][y][y][y][1] = lambda + 2 * mu;
//    coef[z][z][z][z][1] = lambda + 2 * mu;
//
//    coef[x][x][y][y][1] = lambda;
//    coef[y][y][x][x][1] = lambda;
//
//    coef[x][y][x][y][1] = mu;
//    coef[y][x][y][x][1] = mu;
//    coef[x][y][y][x][1] = mu;
//    coef[y][x][x][y][1] = mu;
//
//    coef[x][x][z][z][1] = lambda;
//    coef[z][z][x][x][1] = lambda;
//
//    coef[x][z][x][z][1] = mu;
//    coef[z][x][z][x][1] = mu;
//    coef[x][z][z][x][1] = mu;
//    coef[z][x][x][z][1] = mu;
//
//    coef[z][z][y][y][1] = lambda;
//    coef[y][y][z][z][1] = lambda;
//
//    coef[z][y][z][y][1] = mu;
//    coef[y][z][y][z][1] = mu;
//    coef[z][y][y][z][1] = mu;
//    coef[y][z][z][y][1] = mu;

    set_coef (coef, 0, 1.0, 0.25);
    set_coef (coef, 1, 2.0, 0.25);
    
    ElasticProblem2DOnCellV2<dim> problem (triangulation, coef);

    REPORT problem .solved ();

    uint8_t width_2d_matrix = (dim + 1) * (dim + 1);

    printf("one///////////////////////////////////////////////////////////////////////////\n\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (coef[im][in][jm][jn][0] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", coef[im][in][jm][jn][0]);
            else
                printf("%f   ", coef[im][in][jm][jn][0]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("two///////////////////////////////////////////////////////////////////////////\n\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (coef[im][in][jm][jn][1] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", coef[im][in][jm][jn][1]);
            else
                printf("%f   ", coef[im][in][jm][jn][1]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("mean///////////////////////////////////////////////////////////////////////////\n\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (problem.mean_coefficient[im][in][jm][jn] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", problem.mean_coefficient[im][in][jm][jn]);
            else
                printf("%f   ", problem.mean_coefficient[im][in][jm][jn]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("meta///////////////////////////////////////////////////////////////////////////\n\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (problem.meta_coefficient[im][in][jm][jn] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", problem.meta_coefficient[im][in][jm][jn]);
            else
                printf("%f   ", problem.meta_coefficient[im][in][jm][jn]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("anal meta//////////////////////////////////////////////////////////////////////\n\n");

    typename ElasticProblemSup<dim + 1>::TypeCoef anal_meta;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                    anal_meta[i][j][k][l] .resize (1);

    anal_meta = anal (coef, 0.0, 1.0, 3.0, 4.0);

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (anal_meta[im][in][jm][jn][0] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", anal_meta[im][in][jm][jn][0]);
            else
                printf("%f   ", anal_meta[im][in][jm][jn][0]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("meta dif anal meta//////////////////////////////////////////////////////////////////////\n\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            const double a = std::abs(problem.meta_coefficient[im][in][jm][jn] -
                anal_meta[im][in][jm][jn][0]);

            if (a > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", a);
            else
                printf("%f   ", a);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("phys///////////////////////////////////////////////////////////////////////////\n\n");

    typename ElasticProblemSup<dim + 1>::TypeCoef newcoef;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                {
                    newcoef[i][j][k][l] .resize (1);

                    newcoef[i][j][k][l][0] = problem.meta_coefficient[i][j][k][l];
                };

    newcoef = unphysical_to_physicaly(newcoef);

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (newcoef[im][in][jm][jn][0] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", newcoef[im][in][jm][jn][0]);
            else
                printf("%f   ", newcoef[im][in][jm][jn][0]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    double A = 
        1 - 
        (problem.meta_coefficient[x][x][y][y] * problem.meta_coefficient[y][y][x][x] + 
         problem.meta_coefficient[z][z][x][x] * problem.meta_coefficient[x][x][z][z] + 
         problem.meta_coefficient[y][y][z][z] * problem.meta_coefficient[z][z][y][y]) -
        (problem.meta_coefficient[x][x][y][y] * problem.meta_coefficient[y][y][z][z] * 
         problem.meta_coefficient[z][z][x][x] + problem.meta_coefficient[x][x][z][z] * 
         problem.meta_coefficient[z][z][y][y] * problem.meta_coefficient[y][y][x][x]);

    double B = problem.meta_coefficient[x][x][x][x] / 
        ((1 - problem.meta_coefficient[y][y][z][z] *
              problem.meta_coefficient[z][z][y][y]) /
         (A));

    printf("%f %f \n", B, A);

//    problem .print_result ("output-");
//
///////////////////   // /

};

template <uint8_t dim>
void foo_3 (
        const dealii::Triangulation<dim> &triangulation)
{
    const uint8_t x = 0;
    const uint8_t y = 1;
    const uint8_t z = 2;

    typename ElasticProblemSup<dim + 1>::TypeCoef coef;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                    coef[i][j][k][l] .resize (2);

    typename ElasticProblemSup<dim + 1>::TypeCoef anal_meta;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                    anal_meta[i][j][k][l] .resize (1);

    double y1 = 0.0;
    double y2 = 0.0;
    double p1 = 0.1;
    double p2 = 0.1;

    std::vector<double> errors;

    for (size_t i = 1; i < 10; ++i)
    {
        y1 = i;
        for (size_t j = 2; j < 10; ++j)
        {
            y2 = j;
            p1 = 0.1;
            for (size_t k = 1; k < 10; ++k)
            {
                p1 += 0.025;
                p2 = 0.1;
                for (size_t l = 1; l < 10; ++l)
                {
                    p2 += 0.025;

                    set_coef (coef, 0, y1, p1);
                    set_coef (coef, 1, y2, p2);

                    ElasticProblem2DOnCellV2<dim> problem (triangulation, coef);

                    REPORT problem .solved ();

                    anal_meta = anal (coef, 0.0, 1.0, 3.0, 4.0);

                    double max_err = 0.0;

                    for (size_t m = 0; m < dim+1; ++m)
                        for (size_t n = 0; n < dim+1; ++n)
                            for (size_t o = 0; o < dim+1; ++o)
                                for (size_t p = 0; p < dim+1; ++p)
                                {
                                    double err = 
                                        std::abs(
                                                problem.meta_coefficient[m][n][o][p] -
                                                anal_meta[m][n][o][p][0]);
                                    if (err > max_err)
                                        max_err = err;
                                };
//                    printf("y1=%f, y2=%f, p1=%f, p2=%f, err=%f\n", 
//                            y1, y2, p1, p2, max_err);
                    errors .push_back (max_err);
//    printf("meta///////////////////////////////////////////////////////////////////////////\n\n");
//
//    uint8_t width_2d_matrix = (dim + 1) * :(dim + 1);
//    for (size_t m = 0; m < width_2d_matrix; ++m)
//    {
//        uint8_t im = m / (dim + 1);
//        uint8_t in = m % (dim + 1);
//
//        for (size_t n = 0; n < width_2d_matrix; ++n)
//        {
//            uint8_t jm = n / (dim + 1);
//            uint8_t jn = n % (dim + 1);
//
//            if (problem.meta_coefficient[im][in][jm][jn] > 0.0000001)
//                printf("\x1B[31m%f\x1B[0m   ", problem.meta_coefficient[im][in][jm][jn]);
//            else
//                printf("%f   ", problem.meta_coefficient[im][in][jm][jn]);
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
                };
            };
        };
    };

    for (size_t i = 0; i < errors.size(); ++i)
        if (errors[i] > 0.000001)
            printf("%f\n", errors[i]);
};

template <uint8_t dim>
void set_band (dealii::Triangulation<dim> &triangulation, 
        const double lower, const double top, size_t n_refine)
{
    const double x0 = 0.0;
    const double x1 = lower;
    const double x3 = top;
    const double x4 = 128.0;

    std::vector<dealii::Point< 2 > > v (8);

    v[0][0] = x0; v[0][1] = x0;
    v[1][0] = x1; v[1][1] = x0;
    v[2][0] = x3; v[2][1] = x0;
    v[3][0] = x4; v[3][1] = x0;
    v[4][0] = x0; v[4][1] = x4;
    v[5][0] = x1; v[5][1] = x4;
    v[6][0] = x3; v[6][1] = x4;
    v[7][0] = x4; v[7][1] = x4;

//    v[0][0] = x0; v[0][1] = x0;
//    v[1][0] = x0; v[1][1] = x1;
//    v[2][0] = x0; v[2][1] = x3;
//    v[3][0] = x0; v[3][1] = x4;
//    v[4][0] = x4; v[4][1] = x0;
//    v[5][0] = x4; v[5][1] = x1;
//    v[6][0] = x4; v[6][1] = x3;
//    v[7][0] = x4; v[7][1] = x4;

    std::vector< dealii::CellData< 2 > > c (3, dealii::CellData<2>());

    c[0].vertices[0] = 0;
    c[0].vertices[1] = 1;
    c[0].vertices[2] = 4;
    c[0].vertices[3] = 5;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 5;
    c[1].vertices[3] = 6;
    c[1].material_id = 1;

    c[2].vertices[0] = 2;
    c[2].vertices[1] = 3;
    c[2].vertices[2] = 6;
    c[2].vertices[3] = 7;
    c[2].material_id = 0;

    triangulation .create_triangulation (v, c, dealii::SubCellData());

    triangulation .refine_global (n_refine);
};

template <uint8_t dim>
void set_quadrate (dealii::Triangulation<dim> &triangulation, 
        const double lower, const double top, size_t n_refine)
{
    const double x0 = 0.0;
    const double x1 = lower;
    const double x2 = top;
    const double x3 = 128.0;

//    std::vector< dealii::Point< 2 > > v (8);
//
//    v[0][0] = x0; v[0][1] = x0;
//    v[1][0] = x4; v[1][1] = x0;
//    v[2][0] = x1; v[2][1] = x1;
//    v[3][0] = x3; v[3][1] = x1;
//    v[4][0] = x0; v[4][1] = x4;
//    v[5][0] = x1; v[5][1] = x3;
//    v[6][0] = x3; v[6][1] = x3;
//    v[7][0] = x4; v[7][1] = x4;
//
//    std::vector< dealii::CellData< 2 > > c (5, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 0;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 4;
//    c[1].vertices[3] = 5;
//    c[1].material_id = 0;
//    
//    c[2].vertices[0] = 2;
//    c[2].vertices[1] = 3;
//    c[2].vertices[2] = 5;
//    c[2].vertices[3] = 6;
//    c[2].material_id = 1;
//
//    c[3].vertices[0] = 4;
//    c[3].vertices[1] = 5;
//    c[3].vertices[2] = 7;
//    c[3].vertices[3] = 6;
//    c[3].material_id = 0;
//    
//    c[4].vertices[0] = 1;
//    c[4].vertices[1] = 7;
//    c[4].vertices[2] = 3;
//    c[4].vertices[3] = 6;
//    c[4].material_id = 0;

    std::vector< dealii::Point< 2 > > v (16);

    v[0]  = dealii::Point<dim>(x0, x0);
    v[1]  = dealii::Point<dim>(x1, x0);
    v[2]  = dealii::Point<dim>(x2, x0);
    v[3]  = dealii::Point<dim>(x3, x0);
    v[4]  = dealii::Point<dim>(x0, x1);
    v[5]  = dealii::Point<dim>(x1, x1);
    v[6]  = dealii::Point<dim>(x2, x1);
    v[7]  = dealii::Point<dim>(x3, x1);
    v[8]  = dealii::Point<dim>(x0, x2);
    v[9]  = dealii::Point<dim>(x1, x2);
    v[10] = dealii::Point<dim>(x2, x2);
    v[11] = dealii::Point<dim>(x3, x2);
    v[12] = dealii::Point<dim>(x0, x3);
    v[13] = dealii::Point<dim>(x1, x3);
    v[14] = dealii::Point<dim>(x2, x3);
    v[15] = dealii::Point<dim>(x3, x3);

    std::vector< dealii::CellData< 2 > > c (9, dealii::CellData<2>());

    c[6].vertices[0] = 8;  c[7].vertices[0] = 9;  c[8].vertices[0] = 10;
    c[6].vertices[1] = 9;  c[7].vertices[1] = 10; c[8].vertices[1] = 11;
    c[6].vertices[2] = 12; c[7].vertices[2] = 13; c[8].vertices[2] = 14;
    c[6].vertices[3] = 13; c[7].vertices[3] = 14; c[8].vertices[3] = 15;
    c[6].material_id = 0;  c[7].material_id = 0;  c[8].material_id = 0;

    c[3].vertices[0] = 4;  c[4].vertices[0] = 5;  c[5].vertices[0] = 6;
    c[3].vertices[1] = 5;  c[4].vertices[1] = 6;  c[5].vertices[1] = 7;
    c[3].vertices[2] = 8;  c[4].vertices[2] = 9;  c[5].vertices[2] = 10;
    c[3].vertices[3] = 9;  c[4].vertices[3] = 10; c[5].vertices[3] = 11;
    c[3].material_id = 0;  c[4].material_id = 1;  c[5].material_id = 0;

    c[0].vertices[0] = 0;  c[1].vertices[0] = 1;  c[2].vertices[0] = 2;
    c[0].vertices[1] = 1;  c[1].vertices[1] = 2;  c[2].vertices[1] = 3;
    c[0].vertices[2] = 4;  c[1].vertices[2] = 5;  c[2].vertices[2] = 6;
    c[0].vertices[3] = 5;  c[1].vertices[3] = 6;  c[2].vertices[3] = 7;
    c[0].material_id = 0;  c[1].material_id = 0;  c[2].material_id = 0;


    triangulation .create_triangulation (v, c, dealii::SubCellData());

    triangulation .refine_global (n_refine);
};

template <uint8_t dim>
void set_cross (dealii::Triangulation<dim> &triangulation, 
        const double length, const double width, size_t n_refine)
{
    const double x0 = (0.0);
    const double x1 = (64.0 - length / 2.0);
    const double x2 = (64.0 - width  / 2.0);
    const double x3 = (64.0 + width  / 2.0);
    const double x4 = (64.0 + length / 2.0);
    const double x5 = (128.0);

    std::vector< dealii::Point< 2 > > v (36);

    v[0]  = dealii::Point<dim>(x0, x0);
    v[1]  = dealii::Point<dim>(x1, x0);
    v[2]  = dealii::Point<dim>(x2, x0);
    v[3]  = dealii::Point<dim>(x3, x0);
    v[4]  = dealii::Point<dim>(x4, x0);
    v[5]  = dealii::Point<dim>(x5, x0);

    v[6]  = dealii::Point<dim>(x0, x1);
    v[7]  = dealii::Point<dim>(x1, x1);
    v[8]  = dealii::Point<dim>(x2, x1);
    v[9]  = dealii::Point<dim>(x3, x1);
    v[10] = dealii::Point<dim>(x4, x1);
    v[11] = dealii::Point<dim>(x5, x1);

    v[12] = dealii::Point<dim>(x0, x2);
    v[13] = dealii::Point<dim>(x1, x2);
    v[14] = dealii::Point<dim>(x2, x2);
    v[15] = dealii::Point<dim>(x3, x2);
    v[16] = dealii::Point<dim>(x4, x2);
    v[17] = dealii::Point<dim>(x5, x2);

    v[18] = dealii::Point<dim>(x0, x3);
    v[19] = dealii::Point<dim>(x1, x3);
    v[20] = dealii::Point<dim>(x2, x3);
    v[21] = dealii::Point<dim>(x3, x3);
    v[22] = dealii::Point<dim>(x4, x3);
    v[23] = dealii::Point<dim>(x5, x3);

    v[24] = dealii::Point<dim>(x0, x4);
    v[25] = dealii::Point<dim>(x1, x4);
    v[26] = dealii::Point<dim>(x2, x4);
    v[27] = dealii::Point<dim>(x3, x4);
    v[28] = dealii::Point<dim>(x4, x4);
    v[29] = dealii::Point<dim>(x5, x4);
    
    v[30] = dealii::Point<dim>(x0, x5);
    v[31] = dealii::Point<dim>(x1, x5);
    v[32] = dealii::Point<dim>(x2, x5);
    v[33] = dealii::Point<dim>(x3, x5);
    v[34] = dealii::Point<dim>(x4, x5);
    v[35] = dealii::Point<dim>(x5, x5);

    std::vector< dealii::CellData< 2 > > c (25, dealii::CellData<2>());

c[20].vertices[0]=24; c[21].vertices[0]=25; c[22].vertices[0]=26; c[23].vertices[0]=27; c[24].vertices[0]=28;
c[20].vertices[1]=25; c[21].vertices[1]=26; c[22].vertices[1]=27; c[23].vertices[1]=28; c[24].vertices[1]=29;
c[20].vertices[2]=30; c[21].vertices[2]=31; c[22].vertices[2]=32; c[23].vertices[2]=33; c[24].vertices[2]=34;
c[20].vertices[3]=31; c[21].vertices[3]=32; c[22].vertices[3]=33; c[23].vertices[3]=34; c[24].vertices[3]=35;
c[20].material_id=0;  c[21].material_id=0;  c[22].material_id=0;  c[23].material_id=0;  c[24].material_id=0;

c[15].vertices[0]=18; c[16].vertices[0]=19; c[17].vertices[0]=20; c[18].vertices[0]=21; c[19].vertices[0]=22;
c[15].vertices[1]=19; c[16].vertices[1]=20; c[17].vertices[1]=21; c[18].vertices[1]=22; c[19].vertices[1]=23;
c[15].vertices[2]=24; c[16].vertices[2]=25; c[17].vertices[2]=26; c[18].vertices[2]=27; c[19].vertices[2]=28;
c[15].vertices[3]=25; c[16].vertices[3]=26; c[17].vertices[3]=27; c[18].vertices[3]=28; c[19].vertices[3]=29;
c[15].material_id=0;  c[16].material_id=0;  c[17].material_id=1;  c[18].material_id=0;  c[19].material_id=0;

c[10].vertices[0]=12; c[11].vertices[0]=13; c[12].vertices[0]=14; c[13].vertices[0]=15; c[14].vertices[0]=16;
c[10].vertices[1]=13; c[11].vertices[1]=14; c[12].vertices[1]=15; c[13].vertices[1]=16; c[14].vertices[1]=17;
c[10].vertices[2]=18; c[11].vertices[2]=19; c[12].vertices[2]=20; c[13].vertices[2]=21; c[14].vertices[2]=22;
c[10].vertices[3]=19; c[11].vertices[3]=20; c[12].vertices[3]=21; c[13].vertices[3]=22; c[14].vertices[3]=23;
c[10].material_id=0;  c[11].material_id=1;  c[12].material_id=1;  c[13].material_id=1;  c[14].material_id=0;

c[5].vertices[0]=6;   c[6].vertices[0]=7;   c[7].vertices[0]=8;   c[8].vertices[0]=9;   c[9].vertices[0]=10;
c[5].vertices[1]=7;   c[6].vertices[1]=8;   c[7].vertices[1]=9;   c[8].vertices[1]=10;  c[9].vertices[1]=11;
c[5].vertices[2]=12;  c[6].vertices[2]=13;  c[7].vertices[2]=14;  c[8].vertices[2]=15;  c[9].vertices[2]=16;
c[5].vertices[3]=13;  c[6].vertices[3]=14;  c[7].vertices[3]=15;  c[8].vertices[3]=16;  c[9].vertices[3]=17;
c[5].material_id=0;   c[6].material_id=0;   c[7].material_id=1;   c[8].material_id=0;   c[9].material_id=0;

c[0].vertices[0]=0;   c[1].vertices[0]=1;   c[2].vertices[0]=2;   c[3].vertices[0]=3;   c[4].vertices[0]=4;
c[0].vertices[1]=1;   c[1].vertices[1]=2;   c[2].vertices[1]=3;   c[3].vertices[1]=4;   c[4].vertices[1]=5;
c[0].vertices[2]=6;   c[1].vertices[2]=7;   c[2].vertices[2]=8;   c[3].vertices[2]=9;   c[4].vertices[2]=10;
c[0].vertices[3]=7;   c[1].vertices[3]=8;   c[2].vertices[3]=9;   c[3].vertices[3]=10;  c[4].vertices[3]=11;
c[0].material_id=0;   c[1].material_id=0;   c[2].material_id=0;   c[3].material_id=0;   c[4].material_id=0;

    triangulation .create_triangulation (v, c, dealii::SubCellData());

    if (n_refine > 0)
        triangulation .refine_global (n_refine);

    std::ofstream out ("grid-cross.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (triangulation, out);
};

template <uint8_t dim>
void set_shell(dealii::Triangulation<dim> &triangulation, 
        const double length, const double width, size_t n_refine)
{
    const double dot[7] = 
    {
        (0.0),
        (64.0 - length / 2.0),
        (64.0 - length / 2.0 + width),
        (64.0),
        (64.0 + length / 2.0 - width),
        (64.0 + length / 2.0),
        (128.0)
    };

    std::vector< dealii::Point< 2 > > v (49);

    FOR_I (0, 7)
        FOR_J (0, 7)
        {
            v[i * 7 + j] = dealii::Point<dim>(dot[j], dot[i]);
        };

    std::vector< dealii::CellData< 2 > > c (36, dealii::CellData<2>());

    const size_t material_id[6][6] =
    {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0}
    };

    FOR_I (0, 6)
        FOR_J (0, 6)
        {
            c[i * 6 + j].vertices[0] = i * 7 + j + 0;
            c[i * 6 + j].vertices[1] = i * 7 + j + 1;
            c[i * 6 + j].vertices[2] = i * 7 + j + 7;
            c[i * 6 + j].vertices[3] = i * 7 + j + 8;

            c[i * 6 + j].material_id = material_id[i][j];
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());

    if (n_refine > 0)
        triangulation .refine_global (n_refine);

    std::ofstream out ("grid-cross.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (triangulation, out);
};

void set_circ(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
//    puts("1111111111111111111111111111111111111111111");
    dealii::GridGenerator ::hyper_cube (triangulation, 0, 1);
    triangulation .refine_global (n_refine);
    {
        dealii::Point<2> center (0.5, 0.5);
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 4; ++i)
            {
                midle_p(0) += cell->vertex(i)(0);
                midle_p(1) += cell->vertex(i)(1);
            };
            midle_p(0) /= 4;
            midle_p(1) /= 4;

//            printf("%f %f\n", midle_p(0), midle_p(1));

            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
//                puts("adf");
            }
            else
                cell->set_material_id(0);
        };
    };
};

void set_rectangle(dealii::Triangulation< 2 > &triangulation, 
        const double x1, const double y1,
        const double x2, const double y2, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0, 1);
    triangulation .refine_global (n_refine);
    {
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            uint8_t count = 0;
            for (size_t i = 0; i < 4; ++i)
            {
                if (
                        cell->vertex(i)(0) < (x2 + 1e-10) and
                        cell->vertex(i)(1) < (y2 + 1e-10) and   
                        cell->vertex(i)(0) > (x1 - 1e-10) and
                        cell->vertex(i)(1) > (y1 - 1e-10))   
                    count++;
            };

            if (count >= 3)
            {
                cell->set_material_id(1);
            }
            else
                cell->set_material_id(0);

        };
    };
};

void set_arbirary_tria(dealii::Triangulation< 2 > &triangulation)
{
    std::vector<dealii::Point< 2 > > v (6);

    v[0][0] = 0.0; v[0][1] = 0.0;
    v[1][0] = 2.0; v[1][1] = 0.0;
    v[2][0] = 4.0; v[2][1] = 0.0;
    v[3][0] = 0.0; v[3][1] = 4.0;
    v[4][0] = 2.0; v[4][1] = 4.0;
    v[5][0] = 4.0; v[5][1] = 4.0;

    std::vector< dealii::CellData< 2 > > c (2, dealii::CellData<2>());

    c[0].vertices[0] = 0;
    c[0].vertices[1] = 1;
    c[0].vertices[2] = 3;
    c[0].vertices[3] = 4;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 4;
    c[1].vertices[3] = 5;
    c[1].material_id = 0;

    dealii::GridReordering<2>::reorder_cells(c);

    triangulation .create_triangulation (v, c, dealii::SubCellData());
    triangulation .refine_global (1);

};

void set_grid(dealii::Triangulation< 2 > &triangulation)
{
    std::vector<dealii::Point<2> > v;
    std::vector<dealii::CellData<2> > c;

    CDT cdt;
    std::vector<Vertex_handle> vec_of_vertices;
    std::vector<std::vector<Vertex_handle> > vec_of_domains;

//    double x_begin = -1.0;
//    double x_end = 1.0;
//    double step_by_x = 0.05;
//    double x_current;
//    size_t num_node_x = (x_end - x_begin) / step_by_x + 1;
//
//    for(size_t item_node_x = 0; item_node_x < num_node_x; ++item_node_x)
//    {
//         x_current = x_begin + item_node_x * step_by_x;
//         vec_of_vertices.push_back( cdt.insert( 
//                     Point(x_current*40.0+64.0, 
//                         sqrt(1.0 - x_current * x_current)*40.0+64.0 )) );  
//    }
//
//    for(size_t item_node_x = 1; item_node_x < num_node_x - 1; ++item_node_x)
//    {
//        x_current = x_end - item_node_x * step_by_x;
//        vec_of_vertices.push_back( cdt.insert( 
//                    Point(x_current*40.0+64.0, 
//                        - sqrt(1.0 - x_current * x_current)*40.0+64.0 )) );  
//    }
    
//    vec_of_vertices.push_back(cdt.insert(Point(64.0-32.0, 64.0-64.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(64.0+32.0, 64.0-64.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(64.0+32.0, 64.0+64.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(64.0-32.0, 64.0+64.0) ));

//    vec_of_vertices.push_back(cdt.insert(Point(128.0/3.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0/3.0*2.0,  0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0/3.0*2.0,  128.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0/3.0,  128.0) ));

    Hexagon hexagon(dealii::Point<2>(0.5, 0.5), 0.5/3.141592653589);

    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 1.0/3.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 1.0/3.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 2.0/3.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 2.0/3.0) ));

    // vec_of_vertices.push_back(cdt.insert(Point(0.0, 0.5) ));
    // vec_of_vertices.push_back(cdt.insert(Point(0.5, 0.0) ));
    // vec_of_vertices.push_back(cdt.insert(Point(1.0, 0.5) ));
    // vec_of_vertices.push_back(cdt.insert(Point(0.5, 1.0) ));

    // vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 0.0) ));
    // vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 0.0) ));
    // vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0, 1.0) ));
    // vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 1.0) ));

//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[0](0), hexagon.point[0](1))));
//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[1](0), hexagon.point[1](1))));
//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[2](0), hexagon.point[2](1))));
//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[3](0), hexagon.point[3](1))));
//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[4](0), hexagon.point[4](1))));
//    vec_of_vertices.push_back(cdt.insert(Point(hexagon.point[5](0), hexagon.point[5](1))));

    vec_of_domains.push_back(vec_of_vertices);
    vec_of_vertices.clear();     
    
//    vec_of_vertices.push_back(cdt.insert(Point(0.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0, 128.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(0.0, 128.0) ));
//
//    vec_of_domains.push_back(vec_of_vertices);
//    vec_of_vertices.clear();  
    
  double width_outer_domain = 1.0;
  double height_outer_domain = 1.0;
  
  double begin_x_outer_domain = 0.0;
  double begin_y_outer_domain = 0.0;
  
  size_t num_nodes_by_x = 0;
  size_t num_nodes_by_y = 0;
  
  double end_x_outer_domain = begin_x_outer_domain + width_outer_domain;
  double end_y_outer_domain = begin_y_outer_domain + height_outer_domain;
  

  
  double step_by_x = width_outer_domain / (num_nodes_by_x + 1); 
  double step_by_y = height_outer_domain / (num_nodes_by_y + 1); 
  
  for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain + item_node * step_by_x,
                                                 begin_y_outer_domain ) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain,
                                                 begin_y_outer_domain +  item_node * step_by_y) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain - item_node * step_by_x,
                                                 end_y_outer_domain ) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain,
                                                 end_y_outer_domain -  item_node * step_by_y) ));      
  }
  
    vec_of_domains.push_back(vec_of_vertices);
    vec_of_vertices.clear(); 
    
    for(auto it = vec_of_domains.begin(); it != vec_of_domains.end(); ++it) 
    {
       for(auto vit = it->begin() + 1; vit != it->end(); ++vit)
       {
          cdt.insert_constraint( *(vit - 1), *vit );
       }
       cdt.insert_constraint( *( it->end() - 1), *( it->begin() ) );
    }
    
    
    std::list<Point> list_of_seeds;
    list_of_seeds.push_back(Point(0.5, 0.5));

    std::cout << "Meshing the domain..." << std::endl;
  
    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria(), true);
    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria());
//    CGAL::refine_Delaunay_mesh_2(cdt, Criteria());

    std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
    std::cout << "Number of finite faces: " << cdt.number_of_faces() << std::endl;
    
    size_t mesh_faces_counter = 0;
  
    for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) 
    {
        if(fit->is_in_domain()) 
            ++mesh_faces_counter;
    }
    
    std::cout << "Number of faces in the mesh domain: " << mesh_faces_counter << std::endl;
    
    CDT::Triangle trg;
    
    size_t mat_id; 

    for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
    {
//        puts("!!!!!!!!!!!!!!");
        if(fit->is_in_domain())
        { 
            mat_id = 1;
        }
        else
        {
            mat_id = 0;
        }      
      
        trg = cdt.triangle(fit);
      
        size_t t = v.size()*10;
        size_t indx[7] = {t+0, t+1, t+2, t+3, t+4, t+5, t+6};
      
        double middle_segment01_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()));
        double middle_segment01_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()));
      
        double middle_segment12_x = 0.5 * (CGAL::to_double(trg[1].x()) + CGAL::to_double(trg[2].x()));
        double middle_segment12_y = 0.5 * (CGAL::to_double(trg[1].y()) + CGAL::to_double(trg[2].y()));

        double middle_segment02_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[2].x()));
        double middle_segment02_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[2].y()));
      
        double center_of_mass_x = (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()) +
                                   CGAL::to_double(trg[2].x())) / 3.0;
        double center_of_mass_y = (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()) +
                                   CGAL::to_double(trg[2].y())) / 3.0;
      
        FOR_I(0, v.size())
        {
            if (
                    (fabs(v[i](0) - trg[0].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[0].y()) < 1e-10))
                indx[0] = i;
            if (
                    (fabs(v[i](0) - trg[1].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[1].y()) < 1e-10))
                indx[1] = i;
            if (
                    (fabs(v[i](0) - trg[2].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[2].y()) < 1e-10))
                indx[2] = i;
            if (
                    (fabs(v[i](0) - middle_segment01_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment01_y) < 1e-10))
                indx[3] = i;
            if (
                    (fabs(v[i](0) - middle_segment12_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment12_y) < 1e-10))
                indx[4] = i;
            if (
                    (fabs(v[i](0) - middle_segment02_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment02_y) < 1e-10))
                indx[5] = i;
            if (
                    (fabs(v[i](0) - center_of_mass_x) < 1e-10) and
                    (fabs(v[i](1) - center_of_mass_y) < 1e-10))
                indx[6] = i;
        };

        if (indx[0] == t)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
            indx[0] = v.size()-1;
        }
        if (indx[1] == t+1)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
            indx[1] = v.size()-1;
        }
        if (indx[2] == t+2)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
            indx[2] = v.size()-1;
        }
        if (indx[3] == t+3)
        {
            v.push_back(dealii::Point<2>( middle_segment01_x, middle_segment01_y ));
            indx[3] = v.size()-1;
        }
        if (indx[4] == t+4)
        {
            v.push_back(dealii::Point<2>( middle_segment12_x, middle_segment12_y ));
            indx[4] = v.size()-1;
        }
        if (indx[5] == t+5)
        {
            v.push_back(dealii::Point<2>( middle_segment02_x, middle_segment02_y ));
            indx[5] = v.size()-1;
        }
        if (indx[6] == t+6)
        {
            v.push_back(dealii::Point<2>( center_of_mass_x, center_of_mass_y));
            indx[6] = v.size()-1;
        }

//        printf("%d %d %d %d %d %d %d\n", 
//                indx[0],
//                indx[1],
//                indx[2],
//                indx[3],
//                indx[4],
//                indx[5],
//                indx[6]);

//        v.push_back(dealii::Point<2>(0.0, 0.0));
//        v.push_back(dealii::Point<2>(1.0, 0.0));
//        v.push_back(dealii::Point<2>(2.0, 0.0));
//        v.push_back(dealii::Point<2>(0.0, 2.0));
//        v.push_back(dealii::Point<2>(1.0, 2.0));
//        v.push_back(dealii::Point<2>(2.0, 2.0));

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[0];
            temp_c.vertices[1] = indx[3];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[5];
            temp_c.material_id = mat_id;

//            double mid_x = (
//                    v[indx[0]](0) +
//                    v[indx[3]](0) +
//                    v[indx[6]](0) +
//                    v[indx[5]](0)) / 4.0;
//
//            double mid_y = (
//                    v[indx[0]](1) +
//                    v[indx[3]](1) +
//                    v[indx[6]](1) +
//                    v[indx[5]](1)) / 4.0;
//
//            if (((mid_x < 64.0) and (mid_y < 64.0)) or
//                ((mid_x > 64.0) and (mid_y > 64.0)))
//                temp_c.material_id = 1;
//            else
//                temp_c.material_id = 0;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[1];
            temp_c.vertices[1] = indx[4];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[3];
            temp_c.material_id = mat_id;

//            double mid_x = (
//                    v[indx[1]](0) +
//                    v[indx[4]](0) +
//                    v[indx[6]](0) +
//                    v[indx[3]](0)) / 4.0;
//
//            double mid_y = (
//                    v[indx[1]](1) +
//                    v[indx[4]](1) +
//                    v[indx[6]](1) +
//                    v[indx[3]](1)) / 4.0;
//
//            if (((mid_x < 64.0) and (mid_y < 64.0)) or
//                ((mid_x > 64.0) and (mid_y > 64.0)))
//                temp_c.material_id = 1;
//            else
//                temp_c.material_id = 0;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[2];
            temp_c.vertices[1] = indx[5];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[4];
            temp_c.material_id = mat_id;

//            double mid_x = (
//                    v[indx[2]](0) +
//                    v[indx[5]](0) +
//                    v[indx[6]](0) +
//                    v[indx[4]](0)) / 4.0;
//
//            double mid_y = (
//                    v[indx[2]](1) +
//                    v[indx[5]](1) +
//                    v[indx[6]](1) +
//                    v[indx[4]](1)) / 4.0;
//
//            if (((mid_x < 64.0) and (mid_y < 64.0)) or
//                ((mid_x > 64.0) and (mid_y > 64.0)))
//                temp_c.material_id = 1;
//            else
//                temp_c.material_id = 0;

            c.push_back(temp_c);
        }


    }

//    printf("%d %d\n", v.size(), c.size());
    puts("1");
//    dealii::GridReordering<2>::invert_all_cells_of_negative_grid(v, c);
    dealii::GridReordering<2>::reorder_cells(c);
    puts("2");
    triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
    puts("3");

};

template <size_t num_points>
void set_tria(dealii::Triangulation< 2 > &triangulation, 
        const double points[num_points], 
        const size_t material_id[num_points - 1][num_points - 1])
{

    const size_t num_cells = num_points - 1;

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(points[j], points[i]);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            c[i * num_cells + j].vertices[0] = i * num_points + j + 0;
            c[i * num_cells + j].vertices[1] = i * num_points + j + 1;
            c[i * num_cells + j].vertices[2] = i * num_points + j + num_points;
            c[i * num_cells + j].vertices[3] = i * num_points + j + num_points + 1;

            c[i * num_cells + j].material_id = material_id[i][j];
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

void set_hexagon_grid_pure(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{
    double Ro = len_edge;
    double ro = radius;
    double Ri = Ro * (sqrt(3.0) / 2.0);
    double ri = ro * (sqrt(3.0) / 2.0);

    printf("Radius %f %f\n", ro, ri);

    double a[7] = {0.0, Ri-ri, Ri, Ri+ri, 2.0*Ri, ri, 2.0*Ri - ri};
    double b[15] = {
        0.0, ro/2.0, ro, Ro/2, Ro, 1.5*Ro - ro, 1.5*Ro - ro / 2.0,
        1.5*Ro, 1.5*Ro + ro / 2.0, 1.5*Ro + ro, 2.0*Ro, 2.5*Ro, 3.0*Ro-ro,
        3.0*Ro-ro/2.0, 3.0*Ro};


    std::vector<dealii::Point< 2 > > v (30); //30

//    v[0][0]  = a[0]; v[0][1]  = b[0];
//    v[1][0]  = a[1]; v[1][1]  = b[0];
//    v[2][0]  = a[1]; v[2][1]  = b[1];
//    v[3][0]  = a[0]; v[3][1]  = b[1];
//    v[4][0]  = a[4]; v[4][1]  = b[3];
//    v[5][0]  = a[2]; v[5][1]  = b[4];
//    v[6][0]  = a[2]; v[6][1]  = b[10];
//    v[7][0]  = a[0]; v[7][1]  = b[11];
//    v[8][0]  = a[4]; v[8][1]  = b[11];
//    v[9][0]  = a[0]; v[9][1]  = b[14];
//    v[10][0] = a[2]; v[10][1] = b[14];
//    v[11][0] = a[4]; v[11][1] = b[14];
    

    v[0][0]  = a[0]; v[0][1]  = b[0];
    v[1][0]  = a[1]; v[1][1]  = b[0];
    v[2][0]  = a[2]; v[2][1]  = b[0];
    v[3][0]  = a[3]; v[3][1]  = b[0];
    v[4][0]  = a[4]; v[4][1]  = b[0];

    v[5][0]  = a[1]; v[5][1]  = b[1];
    v[6][0]  = a[3]; v[6][1]  = b[1];

    v[7][0]  = a[2]; v[7][1]  = b[2];

    v[8][0]  = a[0]; v[8][1]  = b[3];
    v[9][0]  = a[4]; v[9][1]  = b[3];

    v[10][0] = a[2]; v[10][1] = b[4];

    v[11][0] = a[0]; v[11][1] = b[5];
    v[12][0] = a[4]; v[12][1] = b[5];

    v[13][0] = a[5]; v[13][1] = b[6];
    v[14][0] = a[6]; v[14][1] = b[6];

    v[15][0] = a[5]; v[15][1] = b[8];
    v[16][0] = a[6]; v[16][1] = b[8];

    v[17][0] = a[0]; v[17][1] = b[9];
    v[18][0] = a[4]; v[18][1] = b[9];

    v[19][0] = a[2]; v[19][1] = b[10];

    v[20][0] = a[0]; v[20][1] = b[11];
    v[21][0] = a[4]; v[21][1] = b[11];

//    v[23][0] = a[4]; v[23][1] = b[8];

    v[22][0] = a[2]; v[22][1] = b[12];

    v[23][0] = a[1]; v[23][1] = b[13];
    v[24][0] = a[3]; v[24][1] = b[13];

    v[25][0] = a[0]; v[25][1] = b[14];
    v[26][0] = a[1]; v[26][1] = b[14];
    v[27][0] = a[2]; v[27][1] = b[14];
    v[28][0] = a[3]; v[28][1] = b[14];
    v[29][0] = a[4]; v[29][1] = b[14];
//
////    v[31][0] = a[4]; v[31][1] = b[8];  // 13

    std::vector< dealii::CellData< 2 > > c (20, dealii::CellData<2>()); //20

//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 5;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 5;
//    c[1].vertices[1] = 1;
//    c[1].vertices[2] = 2;
//    c[1].vertices[3] = 4;
//    c[1].material_id = 0;
//
//    c[2].vertices[0] = 3;
//    c[2].vertices[1] = 5;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 5;
//    c[3].vertices[1] = 4;
//    c[3].vertices[2] = 8;
//    c[3].vertices[3] = 6;
//    c[3].material_id = 0;
//
//    c[4].vertices[0] = 7;
//    c[4].vertices[1] = 6;
//    c[4].vertices[2] = 10;
//    c[4].vertices[3] = 9;
//    c[4].material_id = 0;
//
//    c[5].vertices[0] = 6;
//    c[5].vertices[1] = 8;
//    c[5].vertices[2] = 11;
//    c[5].vertices[3] = 10;
//    c[5].material_id = 0;

//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);
//
////    dealii::GridReordering<2,2>::invert_all_cells_of_negative_grid 
////        (v, c);
//    dealii::GridReordering<2>::reorder_cells(c);
//
//    puts("/////////////////////////////////");
//
//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);



//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;

    c[0].vertices[0] = 1;
    c[0].vertices[1] = 5;
    c[0].vertices[2] = 8;
    c[0].vertices[3] = 0;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 7;
    c[1].vertices[3] = 5;
    c[1].material_id = 1;

    c[2].vertices[0] = 2;
    c[2].vertices[1] = 3;
    c[2].vertices[2] = 6;
    c[2].vertices[3] = 7;
    c[2].material_id = 1;

    c[3].vertices[0] = 3;
    c[3].vertices[1] = 4;
    c[3].vertices[2] = 9;
    c[3].vertices[3] = 6;
    c[3].material_id = 0;

    c[4].vertices[0] = 8;
    c[4].vertices[1] = 5;
    c[4].vertices[2] = 7;
    c[4].vertices[3] = 10;
    c[4].material_id = 0;

    c[5].vertices[0] = 7;
    c[5].vertices[1] = 6;
    c[5].vertices[2] = 9;
    c[5].vertices[3] = 10;
    c[5].material_id = 0;

    c[6].vertices[0] = 8;
    c[6].vertices[1] = 10;
    c[6].vertices[2] = 13;
    c[6].vertices[3] = 11;
    c[6].material_id = 0;

    c[7].vertices[0] = 10;
    c[7].vertices[1] = 9;
    c[7].vertices[2] = 12;
    c[7].vertices[3] = 14;
    c[7].material_id = 0;

    c[8].vertices[0] = 11;
    c[8].vertices[1] = 13;
    c[8].vertices[2] = 15;
    c[8].vertices[3] = 17;
    c[8].material_id = 1;

    c[9].vertices[0] = 13;
    c[9].vertices[1] = 10;
    c[9].vertices[2] = 19;
    c[9].vertices[3] = 15;
    c[9].material_id = 0;

    c[10].vertices[0] = 10;
    c[10].vertices[1] = 14;
    c[10].vertices[2] = 16;
    c[10].vertices[3] = 19;
    c[10].material_id = 0;

    c[11].vertices[0] = 14;
    c[11].vertices[1] = 12;
    c[11].vertices[2] = 18;
    c[11].vertices[3] = 16;
    c[11].material_id = 1;

    c[12].vertices[0] = 17;
    c[12].vertices[1] = 15;
    c[12].vertices[2] = 19;
    c[12].vertices[3] = 20;
    c[12].material_id = 0;

//    c[13].vertices[0] = 15;
//    c[13].vertices[1] = 17;
//    c[13].vertices[2] = 31; //// 31
//    c[13].vertices[3] = 19;
//    c[13].material_id = 1;

    c[13].vertices[0] = 16;
    c[13].vertices[1] = 18;
    c[13].vertices[2] = 21;
    c[13].vertices[3] = 19;
    c[13].material_id = 0;

    c[14].vertices[0] = 20;
    c[14].vertices[1] = 19;
    c[14].vertices[2] = 22;
    c[14].vertices[3] = 23;
    c[14].material_id = 0;

    c[15].vertices[0] = 19;
    c[15].vertices[1] = 21;
    c[15].vertices[2] = 24;
    c[15].vertices[3] = 22;
    c[15].material_id = 0;

    c[16].vertices[0] = 20;
    c[16].vertices[1] = 23;
    c[16].vertices[2] = 26;
    c[16].vertices[3] = 25;
    c[16].material_id = 0;

    c[17].vertices[0] = 23;
    c[17].vertices[1] = 22;
    c[17].vertices[2] = 27;
    c[17].vertices[3] = 26;
    c[17].material_id = 1;

    c[18].vertices[0] = 22;
    c[18].vertices[1] = 24;
    c[18].vertices[2] = 28;
    c[18].vertices[3] = 27;
    c[18].material_id = 1;

    c[19].vertices[0] = 24;
    c[19].vertices[1] = 21;
    c[19].vertices[2] = 29;
    c[19].vertices[3] = 28;
    c[19].material_id = 0;

    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    dealii::GridReordering<2>::reorder_cells(c);
    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
//    triangulation .refine_global (n_ref);

    std::ofstream out ("grid-2.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (triangulation , out);
};

template <size_t num_points>
void set_hexagon_grid(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{

    const size_t num_cells = num_points - 1;
    const double height = len_edge * (sqrt(3.0) / 2.0);
//    printf("%f %f\n", `<args>`);

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * 2.0 * height) / num_points, 
                    (i * 3.0 * len_edge) / num_points);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

        dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            c[cell_number].material_id = 0;

            {
                dealii::Point<2> center (height, 0.0);
                if (Hexagon(center, radius) .include(midle_p))
                    c[cell_number].material_id = 0;
            }

//            {
//                dealii::Point<2> center (3.0 * height, 0.0);
//                if (Hexagon(center, radius) .include(midle_p))
//                    c[cell_number].material_id = 1;
//            }

            {
                dealii::Point<2> center (0.0, 1.5 * len_edge);
                if (Hexagon(center, radius) .include(midle_p))
                    c[cell_number].material_id = 0;
            }

            {
                dealii::Point<2> center (2.0 * height, 1.5 * len_edge);
                if (Hexagon(center, radius) .include(midle_p))
                    c[cell_number].material_id = 0;
            }

//            {
//                dealii::Point<2> center (4.0 * height, 1.5 * len_edge);
//                if (Hexagon(center, radius) .include(midle_p))
//                    c[cell_number].material_id = 1;
//            }

            {
                dealii::Point<2> center (height, 3.0 * len_edge);
                if (Hexagon(center, radius) .include(midle_p))
                    c[cell_number].material_id = 0;
            }

//            {
//                dealii::Point<2> center (3.0 * height, 3.0 * len_edge);
//                if (Hexagon(center, radius) .include(midle_p))
//                    c[cell_number].material_id = 1;
//            }


//            {
//                dealii::Point<2> center (height, 0.0);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (3.0 * height, 0.0);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (0.0, 1.5 * len_edge);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (2.0 * height, 1.5 * len_edge);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (4.0 * height, 1.5 * len_edge);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (height, 3.0 * len_edge);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
//
//            {
//                dealii::Point<2> center (3.0 * height, 3.0 * len_edge);
//                if (center.distance(midle_p) < radius)
//                    c[cell_number].material_id = 1;
//            }
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

template <size_t num_points>
void set_hexagon_brave(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{

    const size_t num_cells = num_points - 1;
    const size_t height = len_edge * (3.0 / 4.0);//(sqrt(3.0) / 2.0);

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * 2.0 * height) / num_points, 
                    (i * 1.0 * len_edge) / num_points);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

        dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            c[cell_number].material_id = 0;

            {
                dealii::Point<2> center (height, 0.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (0.0, len_edge / 2.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (2.0 * height, len_edge / 2.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (height, len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

void set_circ_in_hex(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
    cdbl hight = sqrt(3.0);
    dealii::Point<2> p1(0.0, 0.0);
    dealii::Point<2> p2(2.0, hight);
    dealii::GridGenerator ::hyper_rectangle (triangulation, p1, p2);
    triangulation .refine_global (n_refine);
    {
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 4; ++i)
            {
                midle_p(0) += cell->vertex(i)(0);
                midle_p(1) += cell->vertex(i)(1);
            };
            midle_p(0) /= 4.0;
            midle_p(1) /= 4.0;

            cell->set_material_id(0);
            {
                dealii::Point<2> center (0.5, 0.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (0.0, hight / 2.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (1.0, hight / 2.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (0.5, hight);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (1.5, 0.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (1.5, hight);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
            {
                dealii::Point<2> center (2.0, hight/2.0);
                if (center.distance(midle_p) < radius)
                    cell->set_material_id(1);
            };
        };
    };
};

template <size_t num_points>
void set_line(dealii::Triangulation< 2 > &triangulation, 
        const double hx,
        const double hy,
        const double deriver)
{
    const size_t num_cells = num_points - 1;

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * hx) / (num_points - 1), 
                    (i * hy) / (num_points - 1));
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

            dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            if (midle_p(1) < deriver)
                c[cell_number].material_id = 1;
            else
                c[cell_number].material_id = 0;
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

template <size_t num_points>
void set_hexagon(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{

    const size_t num_cells = num_points - 1;
    const double height = len_edge * (sqrt(3.0) / 2.0);

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(
                    (j * 4.0 * height) / (num_points - 1), 
                    (i * 3.0 * len_edge) / (num_points - 1));
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            const size_t cell_number = i * num_cells + j;
            c[cell_number].vertices[0] = i * num_points + j + 0;
            c[cell_number].vertices[1] = i * num_points + j + 1;
            c[cell_number].vertices[2] = i * num_points + j + num_points;
            c[cell_number].vertices[3] = i * num_points + j + num_points + 1;

        dealii::Point<2> midle_p (0.0, 0.0);

            midle_p(0) = 
                (v[c[cell_number].vertices[1]](0) + 
                v[c[cell_number].vertices[0]](0)) / 2.0;

            midle_p(1) = 
                (v[c[cell_number].vertices[2]](1) + 
                v[c[cell_number].vertices[0]](1)) / 2.0;

            c[cell_number].material_id = 0;

            {
                dealii::Point<2> center (height, 0.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (3.0 * height, 0.0);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (0.0, 1.5 * len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (2.0 * height, 1.5 * len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (4.0 * height, 1.5 * len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (height, 3.0 * len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }

            {
                dealii::Point<2> center (3.0 * height, 3.0 * len_edge);
                if (center.distance(midle_p) < radius)
                    c[cell_number].material_id = 1;
            }
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

template <size_t num_points>
std::array<double, 2> get_fork(const double points[num_points], 
        const size_t material_id[num_points - 1][num_points - 1],
        const double yung_1, const double puasson_1,
        const double yung_2, const double puasson_2)
{
    double length[num_points - 1][2][2] = {0.0};
    double all_length = 0.0;

    const double mu1 = yung_1 / (2 * (1 + puasson_1));
    const double mu2 = yung_2 / (2 * (1 + puasson_2));

    FOR_I(0, num_points - 1)
        FOR_J(0, num_points - 1)
        {
            length[i][material_id[i][j]][0] += points[j + 1] - points[j];
            length[i][material_id[j][i]][1] += points[j + 1] - points[j];
        };

    FOR_I(0, num_points - 1)
        all_length += points[i + 1] - points[i]; 

//    printf("%f\n", all_length);
//
//    printf("\n");
//    FOR_I(0, num_points - 1)
//        printf("%f %f %f %f\n", 
//                length[i][0][0], length[i][1][0],
//                length[i][0][1], length[i][1][1]);

    auto arithmetic = [mu1, mu2] (double l1, double l2) {
        return (mu1 * l1 + mu2 * l2) / (l1 + l2);};

    auto harmonic = [mu1, mu2] (double l1, double l2) {
        return (mu1 * mu2 * (l1 + l2))/(mu1 * l2 + mu2 * l1);};

    double mean[num_points - 1][2] = {0.0};

    FOR_I(0, num_points - 1)
    {
        mean[i][0] = arithmetic(length[i][0][1], length[i][1][1]);
        mean[i][1] = harmonic  (length[i][0][0], length[i][1][0]);
    };

//    printf("\n");
//    FOR_I(0, num_points - 1)
//        printf("%f %f\n", mean[i][0], mean[i][1]);
    
    std::array<double, 2> res = {0.0};

    FOR_I(0, num_points - 1)
    {
        res[0] += (points[i + 1] - points[i]) / mean[i][0];
        res[1] += (points[i + 1] - points[i]) * mean[i][1];
    };

//    printf("\n");
//    printf("%f\n", res[1]);

    res[0] = all_length / res[0];
    res[1] = res[1] / all_length;

//    printf("\n");
//    printf("%f %f\n", res[0], res[1]);
    
    return res;
};

//std::array<std::array<double, 11>, 2> get_fork_ch(
//        const double yung_1, const double puasson_1,
//        const double yung_2, const double puasson_2,
//         const double c)
//{
//    const double E2 = yung_2;
//    const double E1 = yung_1;
//    const double V2 = puasson_2;
//    const double V1 = puasson_1;
//
//    const double M2 = E2 / (2.0 * (1.0 + V2));
//    const double M1 = E1 / (2.0 * (1.0 + V1));
//
//    const double K2 = E2 / (3.0 * (1.0 - 2.0 * V2));
//    const double K1 = E1 / (3.0 * (1.0 - 2.0 * V1));
//
//    const double N2 = 3 - 4 * V2;
//    const double N1 = 3 - 4 * V1;
//
//    std::array<std::array<double, 11>, 2> res; 
//
//    double E11[2], V12[2], M12[2], K23[2], M23[2];
//
//    K23[0] = K1 + 
//
//};

template<uint8_t dim>
dealii::Point<dim, double> get_grad_heat (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> solution,
        uint8_t index_vertex)
{
    dealii::Point<dim, double> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

//    double f1 = cell->vertex(0)(0);
//    double f2 = cell->vertex(1)(0);
//    double f3 = cell->vertex(2)(0);
//    double f4 = cell->vertex(3)(0);

    double f1 = solution(cell->vertex_dof_index (0, 0));
    double f2 = solution(cell->vertex_dof_index (1, 0));
    double f3 = solution(cell->vertex_dof_index (2, 0));
    double f4 = solution(cell->vertex_dof_index (3, 0));

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

    grad(0) = b + d * cell->vertex(index_vertex)(1);
    grad(1) = c + d * cell->vertex(index_vertex)(0);

    return grad;
};

template<uint8_t dim>
std::array<dealii::Point<dim, double>, dim> get_grad_elastic (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> solution,
        uint8_t index_vertex)
{
    std::array<dealii::Point<dim, double>, dim> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    FOR_I(0, dim)
    {
        double f1 = solution(cell->vertex_dof_index (0, i));
        double f2 = solution(cell->vertex_dof_index (1, i));
        double f3 = solution(cell->vertex_dof_index (2, i));
        double f4 = solution(cell->vertex_dof_index (3, i));

        double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
        double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

        grad[i](0) = b + d * cell->vertex(index_vertex)(1);
        grad[i](1) = c + d * cell->vertex(index_vertex)(0);
    };

    return grad;
};

typename ElasticProblemSup<3>::TypeCoef unphysical_to_physicaly (
        typename ElasticProblemSup<3>::TypeCoef unphys)
{
    typename ElasticProblemSup<3>::TypeCoef res;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    res[i][j][k][l] .resize (1);

    double A = 
        unphys[x][x][x][x][0] * unphys[y][y][y][y][0] * unphys[z][z][z][z][0] - 
        unphys[y][y][z][z][0] * unphys[z][z][y][y][0] * unphys[x][x][x][x][0] +
        unphys[x][x][y][y][0] * unphys[y][y][z][z][0] * unphys[z][z][x][x][0] - 
        unphys[y][y][x][x][0] * unphys[x][x][y][y][0] * unphys[z][z][z][z][0] - 
        unphys[y][y][y][y][0] * unphys[x][x][z][z][0] * unphys[z][z][x][x][0] +
        unphys[y][y][x][x][0] * unphys[x][x][z][z][0] * unphys[z][z][y][y][0]; 

//    printf("%f %f %f A = %f\n", 
//            unphys[x][x][x][x][0], 
//            unphys[y][y][y][y][0], 
//            unphys[z][z][z][z][0], 
//            A);

    for (uint8_t i = 0; i < 3; ++i)
    {
        int no_1 = (i + 1) % 3;
        int no_2 = (i + 2) % 3;

        for (uint8_t j = 0; j < 3; ++j)
        {
            int k = (j == no_1) ? no_2 : no_1;

            if (i == j)
                res[i][i][j][j][0] = A;
            else
                res[i][i][j][j][0] = 
                    (unphys[i][i][j][j][0] * unphys[k][k][k][k][0] -
                     unphys[i][i][k][k][0] * unphys[j][j][k][k][0]);

            res[i][i][j][j][0] /= 
                (unphys[no_1][no_1][no_1][no_1][0] * 
                 unphys[no_2][no_2][no_2][no_2][0] - 
                 unphys[no_1][no_1][no_2][no_2][0] * 
                 unphys[no_2][no_2][no_1][no_1][0]);
        };
    };
        
    return res;

};

typename ElasticProblemSup<3>::TypeCoef anal (
        const typename ElasticProblemSup<3>::TypeCoef E,
        const double x1, const double x2, const double x3, const double x4)
{
    typename ElasticProblemSup<3>::TypeCoef meta;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    meta[i][j][k][l] .resize (1);

    const double l1 = (x2 - x1) + (x4 - x3);
    const double l2 = (x3 - x2);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
//                    if (i == j)
                    {
                        const double a = 
                            ((E[i][j][x][x][0] / E[x][x][x][x][0]) * l1 +
                             (E[i][j][x][x][1] / E[x][x][x][x][1]) * l2) /
                            (l1 + l2);
                        const double b = 
                            (E[x][x][x][x][0] * E[x][x][x][x][1] * (l1 + l2)) /
                            (E[x][x][x][x][0] * l2 + E[x][x][x][x][1] * l1);
                        const double c = 
                            ((E[x][x][k][l][0] / E[x][x][x][x][0]) * l1 +
                             (E[x][x][k][l][1] / E[x][x][x][x][1]) * l2) /
                            (l1 + l2);
                        const double d =
                            ((E[i][j][k][l][0] - E[i][j][x][x][0] * 
                              E[x][x][k][l][0] / E[x][x][x][x][0]) * l1 +
                             (E[i][j][k][l][1] - E[i][j][x][x][1] * 
                              E[x][x][k][l][1] / E[x][x][x][x][1]) * l2) /
                            (l1 + l2);

                        meta[i][j][k][l][0] = a * b * c + d;
                    }
//                    else
//                    {
////                        const double delta1 = 
////                            E[x][y][x][y][0] * E[x][z][x][z][0] - 
////                            E[x][y][x][z][0] * E[x][z][x][y][0];
////                        const double delta2 = 
////                            E[x][y][x][y][1] * E[x][z][x][z][1] - 
////                            E[x][y][x][z][1] * E[x][z][x][y][1];
////                        const double a = 
////                            (E[x][y][x][y][0] / delta1) * l1 +
////                            (E[x][y][x][y][1] / delta2) * l2;
////                        const double b = 
////                            (E[x][z][x][z][0] / delta1) * l1 +
////                            (E[x][z][x][z][1] / delta2) * l2;
////                        const double c = 
////                            (E[x][y][x][z][0] / delta1) * l1 +
////                            (E[x][z][x][y][1] / delta2) * l2;
//                        const double a = 
//                            (E[x][y][x][y][0] * l2 + E[x][y][x][y][1] * l1) /
//                            (E[x][y][x][y][0] * E[x][y][x][y][1]);
//                        const double delta = a * a;
//
////                        meta[i][j][k][l][0] = 
////                            (1 / E[x][y][x][y][0] * l1 +
////                             1 / E[x][y][x][y][1] * l2) / delta;
//
//                        meta[i][j][i][l][0] = 
//                            (E[x][y][x][y][0] * E[x][y][x][y][1] * (l1 + l2)) /
//                            (E[x][y][x][y][0] * l2 + E[x][y][x][y][1] * l1);
//                    };
    for (size_t i = 1; i < 3; ++i)
        {
            meta[x][i][x][i][0] = 
                (E[x][y][x][y][0] * E[x][y][x][y][1] * (l1 + l2)) /
                (E[x][y][x][y][0] * l2 + E[x][y][x][y][1] * l1);
            meta[x][i][i][x][0] = meta[x][i][x][i][0];
            meta[i][x][x][i][0] = meta[x][i][x][i][0];
            meta[i][x][i][x][0] = meta[x][i][x][i][0];
        };

    return meta;
};

template<uint8_t dim>
void print_stress(const ElasticProblem2DOnCellV2<2> &problem,
     const typename ElasticProblemSup<dim + 1>::TypeCoef &E,
     cdbl Px, 
     cdbl Py,
     cdbl Pz)
{
    const int8_t x = 0;
    const int8_t y = 1;
    const int8_t z = 2;

    typename ElasticProblemSup<dim + 1>::TypeCoef newcoef;
    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                {
                    newcoef[i][j][k][l] .resize (1);

                    newcoef[i][j][k][l][0] = problem.meta_coefficient[i][j][k][l];
                };
    auto ME = ::unphysical_to_physicaly(newcoef);

    std::vector<std::array<dealii::Point<2>, 3> > 
        grad_elastic_field(problem.system_equations.x.size());

    std::vector<dealii::Point<2> > 
        grad_heat_field(problem.problem_of_torsion_rod.system_equations.x.size());

    {
        typename dealii::DoFHandler<2>::active_cell_iterator cell =
            problem.domain.dof_handler.begin_active();

        typename dealii::DoFHandler<2>::active_cell_iterator endc =
            problem.domain.dof_handler.end();

        std::vector<std::array<uint8_t, 3> > 
            divider(problem.system_equations.x.size());

        for (; cell != endc; ++cell)
        {
            size_t mat_id = cell->material_id();

            FOR_I(0, 4)
                FOR_M(0, 3)
                {
                    std::array<dealii::Point<2>, 2> deform =
                        ::get_grad_elastic<dim> (cell, 
                                problem.solution[m][m], i);  

                    double sigma_xx = 
                        E[x][x][x][x][mat_id] * deform[x](x) +
                        E[x][x][x][y][mat_id] * deform[x](y) +
                        E[x][x][y][x][mat_id] * deform[y](x) +
                        E[x][x][y][y][mat_id] * deform[y](y) +
                        E[x][x][m][m][mat_id];

                    double sigma_xy = 
                        E[x][y][x][x][mat_id] * deform[x](x) +
                        E[x][y][x][y][mat_id] * deform[x](y) +
                        E[x][y][y][x][mat_id] * deform[y](x) +
                        E[x][y][y][y][mat_id] * deform[y](y) +
                        E[x][y][m][m][mat_id];

                    double sigma_yx = 
                        E[y][x][x][x][mat_id] * deform[x](x) +
                        E[y][x][x][y][mat_id] * deform[x](y) +
                        E[y][x][y][x][mat_id] * deform[y](x) +
                        E[y][x][y][y][mat_id] * deform[y](y) +
                        E[y][x][m][m][mat_id];

                    double sigma_yy = 
                        E[y][y][x][x][mat_id] * deform[x](x) +
                        E[y][y][x][y][mat_id] * deform[x](y) +
                        E[y][y][y][x][mat_id] * deform[y](x) +
                        E[y][y][y][y][mat_id] * deform[y](y) +
                        E[y][y][m][m][mat_id];

                    grad_elastic_field[cell->vertex_dof_index(i,0)][m] += 
                        dealii::Point<2>(sigma_xx, sigma_xy);
//                    dealii::Point<2>(
//                            cell->vertex_dof_index(i,0),
//                            cell->vertex_dof_index(i,0));
//                        dealii::Point<2>(
//                                temp[0](0) * 
//                                (cell->material_id() ? 2.0 : 1.0),
//                                (temp[0](1) + temp[1](0)) *
//                                (cell->material_id() ? 2.0 : 1.0) / 
//                                (2.0 * (1.0 + 0.2)));

                    grad_elastic_field[cell->vertex_dof_index(i,1)][m] += 
                        dealii::Point<2>(sigma_yx, sigma_yy);
                    // dealii::Point<2>(
                    //         cell->vertex_dof_index(i,1),
                    //         cell->vertex_dof_index(i,1));
//                        dealii::Point<2>(
//                                (temp[0](1) + temp[1](0)) *
//                                (cell->material_id() ? 2.0 : 1.0) / 
//                                (2.0 * (1.0 + 0.2)),
//                                temp[1](0) * 
//                                (cell->material_id() ? 2.0 : 1.0));

                    divider[cell->vertex_dof_index(i,0)][m] += 1;
                    divider[cell->vertex_dof_index(i,1)][m] += 1;
                };
        };

        FOR_I(0, divider.size())
            FOR_M(0, 3)
            grad_elastic_field[i][m] /= divider[i][m];
        FOR_I(0, divider.size())
            printf("%d %f\n", grad_elastic_field[i][0](0));
    };

    {
        typename dealii::DoFHandler<2>::active_cell_iterator cell =
            problem.problem_of_torsion_rod.domain.dof_handler.begin_active();

        typename dealii::DoFHandler<2>::active_cell_iterator endc =
            problem.problem_of_torsion_rod.domain.dof_handler.end();

        std::vector<uint8_t> 
            divider(problem.problem_of_torsion_rod.system_equations.x.size());

        for (; cell != endc; ++cell)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                grad_heat_field[cell->vertex_dof_index(i,0)] += 
                    ::get_grad_heat<dim> (cell, 
                            problem.problem_of_torsion_rod.solution[0], i) *
                    (cell->material_id() ? 2.0 : 1.0); 

                divider[cell->vertex_dof_index(i,0)] += 1;
            };
        };
        for (size_t i = 0; i < divider.size(); ++i)
            grad_heat_field[i] /= divider[i];
    };

//    for (auto i : {x, y})
//    {
//        char suffix[2] = {'x', 'y'};
//
//        std::string file_name = "grad_";
//        file_name += suffix[i];
//        FOR_J(0, grad_elastic_field.size())
//        {
//            problem.system_equations.x(j) = grad_elastic_field[j](i);
//        };
//        problem.print_result("drad"

    {
        char suffix[3] = {'x', 'y', 'z'};

        dealii::Vector<double> solution(problem.system_equations.x.size());
        dealii::Vector<double> sigma[2];
        sigma[0].reinit(problem.system_equations.x.size());
        sigma[1].reinit(problem.system_equations.x.size());

        // double nu[3] = {- 1.0 / meta_E, + meta_nu_yx / meta_E, + meta_nu_zx / meta_E}; 

        cdbl nu[3] = {
            - Px * 1.0 / ME[x][x][x][x][0] + 
                Py * ME[x][x][y][y][0] / ME[y][y][y][y][0] + Pz * ME[x][x][z][z][0] / ME[z][z][z][z][0],
            - Py * 1.0 / ME[y][y][y][y][0] + 
                Px * ME[y][y][x][x][0] / ME[x][x][x][x][0] + Pz * ME[y][y][z][z][0] / ME[z][z][z][z][0],
            - Pz * 1.0 / ME[z][z][z][z][0] + 
                Px * ME[z][z][x][x][0] / ME[x][x][x][x][0] + Py * ME[z][z][y][y][0] / ME[y][y][y][y][0]};

        printf("NU %f %f %f\n",nu[0], nu[1], nu[2]);

        FOR_I(0, 2)
        {
            FOR_J(0, problem.system_equations.x.size())
                sigma[i](j) = 0.0;
            double integ = 0.0;

            FOR_M(0, 3)
            {

                FOR_J(0, problem.system_equations.x.size())
                {
                    solution(j) = grad_elastic_field[j][m](i);
                    sigma[i](j) += nu[m] * solution(j);
                    integ += nu[m] * solution(j);
                };

                //            integ /= grad_heat_field.size();

                dealii::DataOut<dim> data_out;
                data_out.attach_dof_handler (
                        problem.domain.dof_handler);
                printf("%d\n", solution.size());

                data_out.add_data_vector (solution, "x y");
                data_out.build_patches ();

                std::string file_name = "grad_";
                file_name += suffix[i];
                file_name += "_";
                file_name += suffix[m];
                file_name += suffix[m];
                file_name += ".gpd";

                std::ofstream output (file_name.data());
                data_out.write_gnuplot (output);
            };
            printf("INTEGRAL = %f\n", integ / problem.system_equations.x.size());

            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler (
                    problem.domain.dof_handler);

            data_out.add_data_vector (sigma[i], "x y");
            data_out.build_patches ();

            std::string file_name = "iso/sigma_";
            file_name += suffix[i];
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };
    
        {
            dealii::Vector<double> main_stress(problem.system_equations.x.size());
            dealii::Vector<double> angle(problem.system_equations.x.size());
//            cos[1].reinit(problem.system_equations.x.size());

            std::vector<uint8_t> divider(problem.system_equations.x.size());

//            double max_main_stress[2] = {0.0};
//            double min_main_stress[2] = {0.0};

            typename dealii::DoFHandler<2>::active_cell_iterator cell =
                problem.domain.dof_handler.begin_active();

            typename dealii::DoFHandler<2>::active_cell_iterator endc =
                problem.domain.dof_handler.end();

            for (; cell != endc; ++cell)
            {
                FOR_I(0, 4)
                {
                        sigma[0](cell->vertex_dof_index(i,1)) = 0.0;
                        sigma[1](cell->vertex_dof_index(i,0)) = 0.0;
                    double L1 = 
                        sigma[0](cell->vertex_dof_index(i,0)) +
                        sigma[1](cell->vertex_dof_index(i,1)); 
                    double L2 = 
                        sigma[0](cell->vertex_dof_index(i,0)) *
                        sigma[1](cell->vertex_dof_index(i,1)) -
                        sigma[0](cell->vertex_dof_index(i,1)) *
                        sigma[1](cell->vertex_dof_index(i,0));

//                    main_stress(cell->vertex_dof_index(i,0)) =
                        double str1 = 
                        (L1 - sqrt(L1*L1 - 4.0 * L2)) / 2.0;
//                        printf("s1=%f\n", str1);

//                    main_stress(cell->vertex_dof_index(i,1)) =
                        double str2 = 
                        (L1 + sqrt(L1*L1 - 4.0 * L2)) / 2.0;
//                        printf("s2=%f\n", str2);
//                        printf("%f %f %f\n", 
//                        sigma[0](cell->vertex_dof_index(i,0)),
//                        sigma[1](cell->vertex_dof_index(i,1)),
//                        sigma[0](cell->vertex_dof_index(i,1))
//                                );

                        cdbl id_x = cell->vertex_dof_index(i,x);
                        cdbl id_y = cell->vertex_dof_index(i,y);
                    double angl1 = std::atan(
                            2.0 * sigma[x](id_y) / 
                            (sigma[x](id_x) - sigma[y](id_y))
                            ) / 2.0;
                    double angl2 = angl1 + 3.14159265359 / 2.0; 
                    // double angl2 = std::atan(
                    //         - 2.0 * sigma[y](id_x) / 
                    //         (sigma[x](id_x) - sigma[y](id_y))
                    //         ) / 2.0;

                        // if (sigma[0](cell->vertex_dof_index(i,0)) < sigma[1](cell->vertex_dof_index(i,1)))
                    if (str1 < str2)
                    {
                        double temp = str1;
                        str1 = str2;
                        str2 = temp;

                        temp = angl1;
                        angl1 = angl2;
                        angl2 = temp;
                    };

                    main_stress(cell->vertex_dof_index(i,0)) += str1;
                    main_stress(cell->vertex_dof_index(i,1)) += str2;

                    angle(cell->vertex_dof_index(i,0)) = angl1;// / 3.14159265359 * 180;
                    angle(cell->vertex_dof_index(i,1)) = angl2;// / 3.14159265359 * 180;
//                    cos[0](cell->vertex_dof_index(i,0)) +=
//                        sqrt(1.0 / (1.0 + 
//                                    pow(str1 - 
//                                        sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
//                                    /
//                                    pow(sigma[1](cell->vertex_dof_index(i,0)), 2.0)));
//
//                    cos[0](cell->vertex_dof_index(i,1)) +=
//                        (str1 - 
//                         sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
//                        /
//                        sigma[1](cell->vertex_dof_index(i,0)) 
//                        *
//                        cos[0](cell->vertex_dof_index(i,0));
//
//                    cos[1](cell->vertex_dof_index(i,0)) +=
//                        sqrt(1.0 / (1.0 + 
//                                    pow(str2 - 
//                                        sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
//                                    /
//                                    pow(sigma[1](cell->vertex_dof_index(i,0)), 2.0)));
//
//                    cos[1](cell->vertex_dof_index(i,1)) +=
//                        (str2 - 
//                         sigma[0](cell->vertex_dof_index(i,0)), 2.0) 
//                        /
//                        sigma[1](cell->vertex_dof_index(i,0)) 
//                        *
//                        cos[1](cell->vertex_dof_index(i,0));

                    divider[cell->vertex_dof_index(i,0)] += 1;
                    divider[cell->vertex_dof_index(i,1)] += 1;

//                    if (str1 > max_main_stress[0])
//                        max_main_stress[0] = str1;
//                    if (str2 > max_main_stress[1])
//                        max_main_stress[1] = str2;
//                    if (str1 < min_main_stress[0])
//                        min_main_stress[0] = str1;
//                    if (str2 < max_main_stress[1])
//                        min_main_stress[1] = str2;
                };
            };

            FOR_I(0, divider.size())
            {
                main_stress(i) /= divider[i];
                // angle(i) /= divider[i];
//                cos[0](i) /= divider[i];
//                cos[1](i) /= divider[i];
            };

            {
                dealii::DataOut<dim> data_out;
                data_out.attach_dof_handler (
                        problem.domain.dof_handler);

                data_out.add_data_vector (main_stress, "1 2");
                data_out.build_patches ();

                std::string file_name = "iso/main_stress";
//                file_name += "_2_";
                file_name += std::to_string(name_main_stress);
                file_name += ".gpd";
                printf("DDDDDDDDD %s\n", file_name.data());

                std::ofstream output (file_name.data());
//            FOR_I(0, divider.size())
//            out << 
                data_out.write_gnuplot (output);
            };

//            char suffix[2] = {'1', '2'};
//            FOR_I(0, 2)
//            {
                dealii::DataOut<dim> data_out;
                data_out.attach_dof_handler (
                        problem.domain.dof_handler);

                data_out.add_data_vector (angle, "x y");
                data_out.build_patches ();

                std::string file_name = "iso/angle";
//                file_name += suffix[i];
                file_name += ".gpd";

                std::ofstream output (file_name.data());
                data_out.write_gnuplot (output);
//            };
//                {
//                    FILE *F;
//                    F = fopen("min_max", "w");
//                    fprintf(F, "%f %f\n", min_main_stress[0], max_main_stress[0]);
//                    fprintf(F, "%f %f\n", min_main_stress[1], max_main_stress[1]);
//                    fclose(F);
//                };

        };
    };

    {
        char suffix[3] = {'x', 'y', 'z'};

        dealii::Vector<double> solution(grad_heat_field.size());

        FOR_I(0, dim)
        {
            double integ = 0.0;

            FOR_J(0, grad_heat_field.size())
            {
                solution(j) = grad_heat_field[j](i);
                integ += solution(j);
            };

//            integ /= grad_heat_field.size();
            printf("INTEGRAL = %f\n", integ);

            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler (
                    problem.problem_of_torsion_rod.domain.dof_handler);
            printf("%d\n", solution.size());

            data_out.add_data_vector (
                    solution
//                    problem.problem_of_torsion_rod.heat_flow[i]
                    , "solution");
            data_out.build_patches ();

            std::string file_name = "grad_z";
            file_name += suffix[i];
            file_name += ".gpd";

            std::ofstream output (file_name.data());
            data_out.write_gnuplot (output);
        };
    };


//    dealii::Vector<double> 
//        grad_elastic_field(problem.system_equations.x.size());

};

std::array<double, 11> get_ch(const double yung_1, const double puasson_1,
                                const double yung_2, const double puasson_2,
                                const double c)
{
    const double Ef = yung_2;
    const double Em = yung_1;
    const double Vf = puasson_2;
    const double Vm = puasson_1;

    const double Mf = Ef / (2.0 * (1.0 + Vf));
    const double Mm = Em / (2.0 * (1.0 + Vm));

    const double Kf = Ef / (3.0 * (1.0 - 2.0 * Vf));
    const double Km = Em / (3.0 * (1.0 - 2.0 * Vm));

    const double Nf = 3 - 4 * Vf;
    const double Nm = 3 - 4 * Vm;

    double E11, V12, M12, K23, M23;

    std::array<double, 11> res; 

    {
        const double numerator = 
            (4.0 * c * (1.0 - c) * (Vf - Vm)*(Vf - Vm) * Mm);
        const double denominator = 
            ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0);

        E11 = c * Ef + (1.0 - c) * Em + numerator / denominator;
    };

    {    
        const double numerator = 
        (c * (1.0 - c) * (Vf - Vm) * Mm * (1 / (Km + Mm / 3) - 1 / (Kf - Mf / 3)));
        const double denominator = 
            ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0);

        V12 =  c * Vf + (1.0 - c) * Vm + numerator / denominator;
    };

    {
        M12 =
            Mm * (Mf * (1.0 + c) + Mm * (1.0 - c)) / (Mf * (1.0 - c) + Mm * (1.0 + c));
    };

    {
        const double numerator = 
            c * (Kf - Km + (Mf - Mm) / 3.0) * (Km + 4.0 * Mm / 3.0);
        const double denominator =
            (Km + 4.0 * Mm / 3.0) + (1.0 - c) * (Kf - Km + (Mf - Mm) / 3.0);

        K23 =  
            Km + Mm / 3.0 + numerator / denominator;
    };

    {
        const double aux1 = Mf / Mm;                            
        const double aux1a = (aux1 - 1.0) * c;                
        const double aux1aa = aux1 * Nm + 1.0;                  
        const double aux1ab = aux1aa + aux1a;                    
        const double aux1b = aux1 + Nf;                        
        const double aux1c = (aux1 * Nm - Nf) * c*c*c;            
        const double aux2 = 3.0 * (1.0 - c)*(1.0 - c) * aux1a * aux1b;  
        const double aux3 = aux1 + Nf + aux1c;                  

        const double A1 = aux2;
        const double A2 = (aux1 * Nm + Nf * Nm - aux1c); 
        const double A3 = (Nm * aux1a - aux1aa);
        const double A = A1 + A2 * A3;

        const double B1 = -aux2;
        const double B2 = 0.5 * aux1ab;
        const double B3 = (Nm - 1) * aux1b - 2.0 * aux1c;
        const double B4 = 0.5 * (Nm + 1.0) * aux1a * aux3;
        const double B = 2.0 * (B1 + B2 * B3 + B4);

        const double C1 = aux2;
        const double C2 = aux1ab * aux3;
        const double C = C1 + C2;

        const double D = B*B - 4.0 * A * C;

        M23 =  ( (- sqrt(D) - B) / (2.0 * A)) * Mm;   
    };


    res[0]  = (4.0 * M23 * K23) / (K23 + M23 + 
               4.0 * V12*V12 * M23 * K23 / E11);
    res[1]  = (K23 - M23 - 4.0 * V12*V12 * M23 * K23 / E11) / 
              (K23 + M23 + 4.0 * V12*V12 * M23 * K23 / E11);
    res[2]  = V12 / E11 * res[0];
    res[3]  = res[1];
    res[4]  = res[0];
    res[5]  = res[2];
    res[6]  = V12;
    res[7]  = V12;
    res[8]  = E11;
    res[9]  = M23;
    res[10] = M12;

    return res;
};

std::array<std::array<double, 2>, 12> get_ch_fork(
        const double yung_1, const double puasson_1,
        const double yung_2, const double puasson_2,
        const double c)
{
    const double E1 = yung_2;
    const double E2 = yung_1;
    const double V1 = puasson_2;
    const double V2 = puasson_1;

    const double c1 = c;
    const double c2 = 1 - c;

    const double M1 = E1 / (2.0 * (1.0 + V1));
    const double M2 = E2 / (2.0 * (1.0 + V2));

    const double k1 = E1 / (3.0 * (1.0 - 2.0 * V1));
    const double k2 = E2 / (3.0 * (1.0 - 2.0 * V2));

    const double K1 = k1 + M1 / 3.0;
    const double K2 = k2 + M2 / 3.0;

    double E11[2], V12[2], M12[2], K23[2], M23[2];

    std::array<std::array<double, 2>, 12> res; 

    {
        auto side = [&] (const double mu)
        {
            return (4.0 * c1 * c2 * (V1 - V2) * (V1 - V2)) /
                      (c1 / K2 + c2 / K1 + 1 / mu) +
                   c1 * E1 + c2 * E2;
        };

        E11[0] = side(M2);
        E11[1] = side(M1);
    };

    {    
        auto side = [&] (const double mu)
        {
            return (c1 * c2 * (V1 - V2) * (1 / K2 - 1 / K1)) /
                      (c1 / K2 + c2 / K1 + 1 / mu) +
                   c1 * V1 + c2 * V2;
        };

        const double A = (c1 * c2 * (V1 - V2) * (1 / K2 - 1 / K1));
        const double B = (c1 / K2 + c2 / K1);
        const double C = c1 * V1 + c2 * V2;

        V12[0] = A / (B + 1 / M2) + C;
        V12[1] = A / (B + 1 / M1) + C;

        printf("A=%f B=%f C=%f\n", A, B, C);
//        printf("c1=%f c2=%f (N1-N2)=%f ()=%f\n", c1, c2, (N1-N2), (1 / K2 - 1 / K1));
    };

    {
        M12[0] = M2 + c1 / (1 / (M1 - M2) + c2 / (2.0 * M2));
        M12[1] = M1 + c2 / (1 / (M2 - M1) + c1 / (2.0 * M1));
    };

    {
        K23[0] = K2 + c1 / (1 / (K1 - K2) + c2 / (K2 + M2));
        K23[1] = K1 + c2 / (1 / (K2 - K1) + c1 / (K1 + M1));
    };

    {
        M23[0] = M2 + c1 / 
            (1 / (M1 - M2) + c2 * (K2 + 2.0 * M2) / (2.0 * M2 * (K2 + M2)));
        M23[1] = M1 + c2 / 
            (1 / (M2 - M1) + c1 * (K1 + 2.0 * M1) / (2.0 * M1 * (K1 + M1)));
    };

    FOR_I(0,2)
    {
        res[0][i]  = (4.0 * M23[i] * K23[i]) / (K23[i] + M23[i] + 
                4.0 * V12[i]*V12[i] * M23[i] * K23[i] / E11[i]);
        res[1][i]  = 
            (K23[i] - M23[i] - 4.0 * V12[i]*V12[i] * M23[i] * K23[i] / E11[i]) / 
            (K23[i] + M23[i] + 4.0 * V12[i]*V12[i] * M23[i] * K23[i] / E11[i]);
        res[2][i]  = V12[i] / E11[i] * res[0][i];
        res[3][i]  = res[1][i];
        res[4][i]  = res[0][i];
        res[5][i]  = res[2][i];
        res[6][i]  = V12[i];
        res[7][i]  = V12[i];
        res[8][i]  = E11[i];
        res[9][i]  = M23[i];
        res[10][i] = M12[i];
        res[11][i] = K23[i];
    };

    return res;
};

dbl oMega = 0.0;

template<size_t size>
void print_tensor(const typename ElasticProblemSup<3>::TypeCoef &tensor)
{
    const size_t width = static_cast<size_t>(sqrt(size));

    for (size_t i = 0; i < width; ++i)
    {
        auto ind = to2D<size>(i);
        uint8_t im = ind[0];
        uint8_t in = ind[1];

        for (size_t j = 0; j < width; ++j)
        {
            auto jnd = to2D<size>(j);
            uint8_t jm = jnd[0];
            uint8_t jn = jnd[1];

            if (fabs(tensor[im][in][jm][jn][0]) > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", 
                        tensor[im][in][jm][jn][0]);
            else
                printf("%f   ", 
                        tensor[im][in][jm][jn][0]);
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

    printf("\n");
};

template <uint8_t dim>
std::array<double, 12> solved (dealii::Triangulation<dim> &triangulation,
        const double yung_1, const double puasson_1,
        const double yung_2, const double puasson_2)
{
    const uint8_t x = 0;
    const uint8_t y = 1;
    const uint8_t z = 2;
//
    typename ElasticProblemSup<dim + 1>::TypeCoef coef;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                    coef[i][j][k][l] .resize (2);


    ::set_coef (coef, 0, yung_1, puasson_1);
    ::set_coef (coef, 1, yung_2, puasson_2);
    
    ElasticProblem2DOnCellV2<dim> problem (triangulation, coef);

    puts("1");
    REPORT problem .solved ();

//    {
//        typename dealii::DoFHandler<2>::active_cell_iterator cell =
//            problem.problem_of_torsion_rod.domain.dof_handler.begin_active();
//
//        typename dealii::DoFHandler<2>::active_cell_iterator endc =
//            problem.problem_of_torsion_rod.domain.dof_handler.end();
//
//        std::vector<uint8_t> 
//            divider(problem.problem_of_torsion_rod.solution[0].size());
//
//        for (; cell != endc; ++cell)
//        {
//            for (size_t i = 0; i < 4; ++i)
//            {
//                problem.problem_of_torsion_rod.
//                    solution[0](cell->vertex_dof_index(i,0)) = 
//                    0.000004*cell->vertex(i)(0)*cell->vertex(i)(0);
//
//                divider[cell->vertex_dof_index(i,0)] += 1;
//            };
//        };
////        for (size_t i = 0; i < divider.size(); ++i)
////                problem.problem_of_torsion_rod.
////                    solution[0](i) /= divider[i];
//    };

    puts("2");
    problem .print_result (std::string("res_"));

//
//    
//
    uint8_t width_2d_matrix = (dim + 1) * (dim + 1);
//
   for (size_t i = 0; i < width_2d_matrix; ++i)
   {
       uint8_t im = i / (dim + 1);
       uint8_t in = i % (dim + 1);

       for (size_t j = 0; j < width_2d_matrix; ++j)
       {
           uint8_t jm = j / (dim + 1);
           uint8_t jn = j % (dim + 1);

           if (coef[im][in][jm][jn][1] > 0.0000001)
               printf("\x1B[31m%f\x1B[0m   ", 
                       coef[im][in][jm][jn][0]);
           else
               printf("%f   ", 
                       coef[im][in][jm][jn][0]);
       };
       for (size_t i = 0; i < 2; ++i)
           printf("\n");
   };

           printf("\n");
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / (dim + 1);
//        uint8_t in = i % (dim + 1);
//
//        for (size_t j = 0; j < width_2d_matrix; ++j)
//        {
//            uint8_t jm = j / (dim + 1);
//            uint8_t jn = j % (dim + 1);
//
//            if (coef[im][in][jm][jn][0] > 0.0000001)
//                printf("\x1B[31m%f\x1B[0m   ", 
//                        coef[im][in][jm][jn][1]);
//            else
//                printf("%f   ", 
//                        coef[im][in][jm][jn][1]);
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
////
//            printf("\n");
//    std::array<double, 2> meta1;
//    return meta1;

   for (size_t i = 0; i < width_2d_matrix; ++i)
   {
       uint8_t im = i / (dim + 1);
       uint8_t in = i % (dim + 1);

       for (size_t j = 0; j < width_2d_matrix; ++j)
       {
           uint8_t jm = j / (dim + 1);
           uint8_t jn = j % (dim + 1);

           if (coef[im][in][jm][jn][0] > 0.0000001)
               printf("\x1B[31m%f\x1B[0m   ", 
                       oMega * coef[im][in][jm][jn][0] +
                       (1.0 - oMega) * coef[im][in][jm][jn][1]);
           else
               printf("%f   ", 
                       coef[im][in][jm][jn][0]);
       };
       for (size_t i = 0; i < 2; ++i)
           printf("\n");
   };
            printf("\n");

   for (size_t i = 0; i < width_2d_matrix; ++i)
   {
       uint8_t im = i / (dim + 1);
       uint8_t in = i % (dim + 1);

       for (size_t j = 0; j < width_2d_matrix; ++j)
       {
           uint8_t jm = j / (dim + 1);
           uint8_t jn = j % (dim + 1);

           if (coef[im][in][jm][jn][0] > 0.0000001)
               printf("\x1B[31m%f\x1B[0m   ",
                       (coef[im][in][jm][jn][0] * coef[im][in][jm][jn][1])/
                       (oMega * coef[im][in][jm][jn][1] +
                       (1.0 - oMega) * coef[im][in][jm][jn][0]));
           else
               printf("%f   ", 
                       coef[im][in][jm][jn][0]);
       };
       for (size_t i = 0; i < 2; ++i)
           printf("\n");
   };
            printf("\n");

    for (size_t i = 0; i < width_2d_matrix; ++i)
    {
        uint8_t im = i / (dim + 1);
        uint8_t in = i % (dim + 1);

        for (size_t j = 0; j < width_2d_matrix; ++j)
        {
            uint8_t jm = j / (dim + 1);
            uint8_t jn = j % (dim + 1);

            if (problem.mean_coefficient[im][in][jm][jn] > 0.0000001)
                printf("\x1B[31m%f\x1B[0m   ", 
                        fabs(problem.meta_coefficient[im][in][jm][jn]));
            else
                printf("%f   ", 
                        fabs(problem.meta_coefficient[im][in][jm][jn]));
        };
        for (size_t i = 0; i < 2; ++i)
            printf("\n");
    };

            printf("\n");

//    for (size_t i = 0; i < 6; ++i)
//    {
//        auto ind = to2D(i);
//        uint8_t im = ind[0];
//        uint8_t in = ind[1];
//
//        for (size_t j = 0; j < 6; ++j)
//        {
//            auto jnd = to2D(j);
//            uint8_t jm = jnd[0];
//            uint8_t jn = jnd[1];
//
//            if (fabs(problem.mean_coefficient[im][in][jm][jn]) > 0.0000001)
//                printf("\x1B[31m%f\x1B[0m   ", 
//                        fabs(problem.meta_coefficient[im][in][jm][jn]));
//            else
//                printf("%f   ", 
//                        fabs(problem.meta_coefficient[im][in][jm][jn]));
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
//
//            printf("\n");

    const double angl = (3.14159265359 / 8.0);

    typename ElasticProblemSup<dim + 1>::TypeCoef newcoef;
    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                {
                    newcoef[i][j][k][l] .resize (1);

                    newcoef[i][j][k][l][0] = problem.meta_coefficient[i][j][k][l];
                };

    typename ElasticProblemSup<dim + 1>::TypeCoef S;
    typename ElasticProblemSup<dim + 1>::TypeCoef E;

    for (size_t i = 0; i < dim+1; ++i)
        for (size_t j = 0; j < dim+1; ++j)
            for (size_t k = 0; k < dim+1; ++k)
                for (size_t l = 0; l < dim+1; ++l)
                {
                    S[i][j][k][l] .resize (1);
                    E[i][j][k][l] .resize (1);
                };

    print_tensor<6*6>(newcoef);
    S = unphysical_to_compliance<6*6>(newcoef);
    print_tensor<6*6>(S);
    E = matrix_multiplication(newcoef, S);
    print_tensor<6*6>(E);

   for (size_t i = 0; i < width_2d_matrix; ++i)
   {
       uint8_t im = i / (dim + 1);
       uint8_t in = i % (dim + 1);

       for (size_t j = 0; j < width_2d_matrix; ++j)
       {
           uint8_t jm = j / (dim + 1);
           uint8_t jn = j % (dim + 1);

           if (coef[im][in][jm][jn][1] > 0.0000001)
               printf("\x1B[31m%f\x1B[0m   ", 
                       newcoef[im][in][jm][jn][0]);
           else
               printf("%f   ", 
                       newcoef[im][in][jm][jn][0]);
       };
       for (size_t i = 0; i < 2; ++i)
           printf("\n");
   };

           printf("\n");
//    turn(newcoef, angl);
//
//    print_tensor<6*6>(newcoef);
//    S = unphysical_to_compliance<6*6>(newcoef);
//    print_tensor<6*6>(S);
//    E = matrix_multiplication(newcoef, S);
//    print_tensor<6*6>(E);
//

//    check_tetragonality(newcoef, S);

//    printf("C %f S %f\n", newcoef[);

    newcoef = ::unphysical_to_physicaly(newcoef);

    print_stress<2>(problem, coef, 0.0, 0.0, 1.0);
            // newcoef[0][0][0][0][0],
            // newcoef[0][0][1][1][0],
            // newcoef[0][0][2][2][0]);
//    printf("PPPPPPPPPP %f %f %f\n", 
//            newcoef[0][0][0][0][0],
//            newcoef[0][0][1][1][0],
//            newcoef[0][0][2][2][0]);
//    {
//        FILE *F;
//        F = fopen("turn.gpd", "w");
//        fclose(F);
//    };
//
//
//    FOR_I(0, 2)
//    {
//
////    for (size_t i = 0; i < 6; ++i)
////    {
////        auto ind = to2D(i);
////        uint8_t im = ind[0];
////        uint8_t in = ind[1];
////
////        for (size_t j = 0; j < 6; ++j)
////        {
////            auto jnd = to2D(j);
////            uint8_t jm = jnd[0];
////            uint8_t jn = jnd[1];
////
////            if (fabs(S[im][in][jm][jn][0]) > 0.0000001)
////                printf("\x1B[31m%f\x1B[0m   ", S[im][in][jm][jn][0]);
////            else
////                printf("%f   ", S[im][in][jm][jn][0]);
////        };
////        for (size_t i = 0; i < 2; ++i)
////            printf("\n");
////    };
////
////            printf("\n");
////
////    for (size_t i = 0; i < 6; ++i)
////    {
////        auto ind = to2D(i);
////        uint8_t im = ind[0];
////        uint8_t in = ind[1];
////
////        for (size_t j = 0; j < 6; ++j)
////        {
////            auto jnd = to2D(j);
////            uint8_t jm = jnd[0];
////            uint8_t jn = jnd[1];
////
////            if (fabs(E[im][in][jm][jn][0]) > 0.0000001)
////                printf("\x1B[31m%f\x1B[0m   ", E[im][in][jm][jn][0]);
////            else
////                printf("%f   ", E[im][in][jm][jn][0]);
////        };
////        for (size_t i = 0; i < 2; ++i)
////            printf("\n");
////    };
//
//      {
//            FILE *F;
//F = fopen("turn.gpd", "a");
////            printf("%f %d %f\n",(3.14159265359 / 2000.0) * i, i, newcoef[0][0][0][1][0]);
//            fprintf(F,"%f %f %f %f %f\n",
//angl * i, newcoef[0][0][0][1][0],
//problem.meta_coefficient[0][1][0][1],
//problem.meta_coefficient[0][0][0][0] - problem.meta_coefficient[0][0][1][1],
//(pow(cos(angl*i), 3) * sin(angl*i) - pow(sin(angl*i), 3) * cos(angl*i))*
//(- 2.0 * problem.meta_coefficient[0][1][0][1]+
//(problem.meta_coefficient[0][0][0][0] - problem.meta_coefficient[0][0][1][1])));
//fclose(F);
//    turn(newcoef, angl);
//};
////    turn(newcoef, 3.14159265359 / 2.0);
////    for (size_t i = 0; i < 6; ++i)
////    {
////        auto ind = to2D(i);
////        uint8_t im = ind[0];
////        uint8_t in = ind[1];
////
////        for (size_t j = 0; j < 6; ++j)
////        {
////            auto jnd = to2D(j);
////            uint8_t jm = jnd[0];
////            uint8_t jn = jnd[1];
////
////            if (newcoef[im][in][jm][jn][0] > 0.0000001)
////                printf("\x1B[31m%f\x1B[0m   ", newcoef[im][in][jm][jn][0]);
////            else
////                printf("%f   ", newcoef[im][in][jm][jn][0]);
////        };
////        for (size_t i = 0; i < 2; ++i)
////            printf("\n");
////    };
////
////            printf("\n");
//    };
//    turn(newcoef, angl);
//    print_tensor<9*9>(newcoef);
    const double W = newcoef[0][0][0][1][0];
//    newcoef = ::unphysical_to_physicaly(newcoef);
//    print_tensor<9*9>(newcoef);
//
//    typename ElasticProblemSup<dim + 1>::TypeCoef anal;
//    for (size_t i = 0; i < dim+1; ++i)
//        for (size_t j = 0; j < dim+1; ++j)
//            for (size_t k = 0; k < dim+1; ++k)
//                for (size_t l = 0; l < dim+1; ++l)
//                {
//                    anal[i][j][k][l] .resize (2);
//                };
//
//    double delta1 = 
//        coef[x][y][x][y][0] * coef[x][z][x][z][0] - 
//        coef[x][y][x][z][0] * coef[x][y][x][z][0];
//    double delta2 = 
//        coef[x][y][x][y][1] * coef[x][z][x][z][1] - 
//        coef[x][y][x][z][1] * coef[x][y][x][z][1];
//
//    printf("%f %f\n", delta1, delta2);
//
////    double S1 = lower
//
//    for (size_t i = 0; i < dim+1; ++i)
//        for (size_t j = 0; j < dim+1; ++j)
//            for (size_t k = 0; k < dim+1; ++k)
//                for (size_t l = 0; l < dim+1; ++l)
//                {
////                    coef[i][j][k][l][0] /= delta1;
////                    coef[i][j][k][l][1] /= delta2;
//                    anal[i][j][k][l][0] = 
//                        ((coef[i][j][k][l][0] / delta1) * 2.0 +
//                         (coef[i][j][k][l][1] / delta2) * 1.0) / 3.0; 
//                };
//
//    double delta3 = 
//        anal[x][y][x][y][0] * anal[x][z][x][z][0] - 
//        anal[x][y][x][z][0] * anal[x][y][x][z][0];
//
//    printf("%f %f %f %f\n", delta3,
//            anal[x][y][x][y][0], anal[x][z][x][z][0], anal[x][y][x][z][0]);
//
////    for (auto i : {0, 1, 2})
////        for (auto j : {0, 1, 2})
////        {
////            anal[x][i][x][j][0] /= delta3;
////            anal[i][x][x][j][0] /= delta3;
////            anal[x][i][j][x][0] /= delta3;
////            anal[i][x][j][x][0] /= delta3;
////            anal[x][x][i][j][0] /= delta3;
////            anal[i][j][x][x][0] /= delta3;
////        };
//    for (size_t i = 0; i < dim+1; ++i)
//        for (size_t j = 0; j < dim+1; ++j)
//            for (size_t k = 0; k < dim+1; ++k)
//                for (size_t l = 0; l < dim+1; ++l)
//                    anal[i][j][k][l][0] /= delta3;
//
//    for (size_t i = 0; i < dim+1; ++i)
//        for (size_t j = 0; j < dim+1; ++j)
//            for (size_t k = 0; k < dim+1; ++k)
//                for (size_t l = 0; l < dim+1; ++l)
//                {
//                    double a = 
//                        ((coef[i][j][x][x][0] / coef[x][x][x][x][0]) * 2.0 +
//                         (coef[i][j][x][x][1] / coef[x][x][x][x][1]) * 1.0) / 3.0; 
//                    double b = 
//                        ((coef[x][x][i][j][0] / coef[x][x][x][x][0]) * 2.0 +
//                         (coef[x][x][i][j][1] / coef[x][x][x][x][1]) * 1.0) / 3.0; 
//                    double c = 
//                        (
//                         (
//                          coef[i][j][k][l][0] - 
//                          (coef[x][x][i][j][0] * coef[i][j][x][x][0] / 
//                           coef[x][x][x][x][0])
//                         ) * 2.0 +
//                         (
//                          coef[i][j][k][l][1] - 
//                          (coef[x][x][i][j][1] * coef[i][j][x][x][1] /
//                          coef[x][x][x][x][1])
//                         ) * 1.0
//                        ) / 3.0; 
//                    double d = 
//                        (coef[x][x][x][x][0] * coef[x][x][x][x][1] * 3.0) /
//                        (coef[x][x][x][x][0] * 1.0 + coef[x][x][x][x][1] * 2.0);
//                    anal[i][j][k][l][1] = a * b * d + c;
////                    printf("%d %d %d %d %f %f %f %f\n", i, j, k, l, a, b, c, d);
//                };
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / (dim + 1);
//        uint8_t in = i % (dim + 1);
//
//        for (size_t j = 0; j < width_2d_matrix; ++j)
//        {
//            uint8_t jm = j / (dim + 1);
//            uint8_t jn = j % (dim + 1);
//
//            if (anal[im][in][jm][jn][0] > 0.0000001)
//            {
//                if ((im == x) or (in == x) or (jm == x) or (jn == x)) then
//                {
//                    printf("\x1B[31m%f\x1B[0m   ", 
//                            anal[im][in][jm][jn][0]);
//                }
//                else 
//                {
//                    printf("\x1B[31m%f\x1B[0m   ", 
//                            anal[im][in][jm][jn][1]);
//                }
//            }
//            else
//                printf("%f   ", 
//                        anal[im][in][jm][jn][0]);
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
//
//            printf("\n");
//
//
//    FILE* F;
//    F = fopen ("E.gpd", "a");
//    fprintf(F, "%f ", glob_i * glob_i);
//    for (auto i : {x, y, z}) 
//        for (auto j : {x, y, z}) 
//            for (auto k : {x, y, z}) 
//                for (auto l : {x, y, z}) 
//                {
//                    fprintf(F, "%f ", (problem.meta_coefficient[i][j][k][l]));
//                };
//    fprintf(F, "\n");
//    fclose(F);
//    printf("%f\n", 
//            (problem.meta_coefficient[0][0][0][0] -
//             problem.meta_coefficient[0][0][1][1]) / 2);

    std::array<double, 12> meta1 = {{0.0}};
    meta1[0]  = newcoef[0][0][0][0][0];
//    puts("dfvdfgv");
    meta1[1]  = newcoef[0][0][1][1][0];
    meta1[2]  = 
//        problem.meta_coefficient[0][0][2][2]/
//        (problem.meta_coefficient[0][0][0][0]+
//        problem.meta_coefficient[0][0][1][1]);
        newcoef[0][0][2][2][0];
    meta1[3]  = newcoef[1][1][0][0][0];
    meta1[4]  = newcoef[1][1][1][1][0];
    meta1[5]  = newcoef[1][1][2][2][0];
    meta1[6]  = newcoef[2][2][0][0][0];
    meta1[7]  = newcoef[2][2][1][1][0];
    meta1[8]  = 
//        problem.meta_coefficient[2][2][2][2]-
//        pow(problem.meta_coefficient[0][0][2][2], 2.0) * 2.0 /
//        (problem.meta_coefficient[0][0][0][0] +
//        problem.meta_coefficient[0][0][1][1]);
newcoef[2][2][2][2][0];
    meta1[9]  = //problem.meta_coefficient[0][0][0][0] -
             //problem.meta_coefficient[0][0][1][1]) / 2;
        problem.meta_coefficient[0][1][0][1];
    meta1[10] = problem.meta_coefficient[0][2][0][2];

    double K23 = (problem.meta_coefficient[0][0][0][0] +
             problem.meta_coefficient[0][0][1][1]) / 2;

    double a = (problem.meta_coefficient[0][0][0][0] -
             problem.meta_coefficient[0][0][1][1]) / 2;

    meta1[11] = W;//a;

    printf("%f %f ERROR=%f\n", a, meta1[9], (a - meta1[9]) / meta1[9]);

    double b = 
        (problem.meta_coefficient[2][2][2][2] *
        problem.meta_coefficient[0][0][1][1] -
        pow(problem.meta_coefficient[0][0][2][2], 2.0)) /
        (problem.meta_coefficient[2][2][2][2] *
        problem.meta_coefficient[1][1][1][1] -
        pow(problem.meta_coefficient[0][0][2][2], 2.0));

//    printf("%f %f\n", b, newcoef[0][0][1][1][0]);
//    printf("%f %f %f\n", meta1[6], 
//
//        problem.meta_coefficient[0][0][2][2]/
//        (problem.meta_coefficient[0][0][0][0]+
//        problem.meta_coefficient[0][0][1][1]),
//
//        problem.meta_coefficient[1][1][2][2]/
//        (problem.meta_coefficient[1][1][1][1]+
//        problem.meta_coefficient[1][1][0][0]));


    return meta1;

};

int main(int argc, char *argv[])
{
//    int x = 0;
//    for (auto i : {0, 1, 2})
//        printf("%d\n", i);
//    for (auto i : {"раз", "два"})
//        printf("%s\n", i);
//    enum {x, y, z};
//    for (auto i : {x, y, z, x})
//        printf("%d\n", i);
//    return 0;
    std::vector<std::array<double, 12>> meta;

//    {
//        dealii::Triangulation<2> tria;
//
//        ::set_quadrate <2> (tria, 24.0, 104.0, 2);
//
////        auto res = ::solved<2>(tria, 100.0, 0.25, 1.0, 0.25);
//        meta .push_back (::solved<2>(tria, 100.0, 0.25, 1.0, 0.25));
////        printf("%f %f\n", res[0], res[1]);
//    };
//                const double length = 128.0 / 6.0 * 4.0;
//                const double width  = 128.0 / 6.0;
//
//                const double dot[7] = 
//                {
//                    (0.0),
//                    (64.0 - length / 2.0),
//                    (64.0 - length / 2.0 + width),
//                    (64.0),
//                    (64.0 + length / 2.0 - width),
//                    (64.0 + length / 2.0),
//                    (128.0)
//                };

//                const double dot[8] = 
//                {
//                    (0.0),
//                    (128.0/ 7.0 * 1),
//                    (128.0/ 7.0 * 2),
//                    (128.0/ 7.0 * 3),
//                    (128.0/ 7.0 * 4),
//                    (128.0/ 7.0 * 5),
//                    (128.0/ 7.0 * 6),
//                    (128.0)
//                };

//    int dim = 2;
//    uint8_t width_2d_matrix = (3) * (3);
//
//    typename ElasticProblemSup<3>::TypeCoef coef;
//
//    for (size_t i = 0; i < 3; ++i)
//        for (size_t j = 0; j < 3; ++j)
//            for (size_t k = 0; k < 3; ++k)
//                for (size_t l = 0; l < 3; ++l)
//                    coef[i][j][k][l] .resize (1);
//
//    ::set_coef (coef, 0, 1.0, 0.35);
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / (dim + 1);
//        uint8_t in = i % (dim + 1);
//
//        for (size_t j = 0; j < width_2d_matrix; ++j)
//        {
//            uint8_t jm = j / (dim + 1);
//            uint8_t jn = j % (dim + 1);
//
//            if (coef[im][in][jm][jn][0] > 0.0000001)
//                printf("\x1B[31m%f\x1B[0m   ", 
//                        coef[im][in][jm][jn][0]);
//            else
//                printf("%f   ", 
//                        coef[im][in][jm][jn][0]);
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
//
//            printf("\n");
//
//    typename ElasticProblemSup<3>::TypeCoef newcoef;
//    for (size_t i = 0; i < 3; ++i)
//        for (size_t j = 0; j < 3; ++j)
//            for (size_t k = 0; k < 3; ++k)
//                for (size_t l = 0; l < 3; ++l)
//                {
//                    newcoef[i][j][k][l] .resize (1);
//
//                    newcoef[i][j][k][l][0] = coef[i][j][k][l][0];
//                };
//
//    newcoef = ::unphysical_to_physicaly(newcoef);
//
//    for (size_t i = 0; i < width_2d_matrix; ++i)
//    {
//        uint8_t im = i / (dim + 1);
//        uint8_t in = i % (dim + 1);
//
//        for (size_t j = 0; j < width_2d_matrix; ++j)
//        {
//            uint8_t jm = j / (dim + 1);
//            uint8_t jn = j % (dim + 1);
//
//            if (newcoef[im][in][jm][jn][0] > 0.0000001)
//                printf("\x1B[31m%f\x1B[0m   ", newcoef[im][in][jm][jn][0]);
//            else
//                printf("%f   ", newcoef[im][in][jm][jn][0]);
//        };
//        for (size_t i = 0; i < 2; ++i)
//            printf("\n");
//    };
//
//    return  0;

    const size_t material_id_for_quadrate[4][4] =
    {
        {0, 0, 0, 0},
        {0, 1, 1, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 0}
    };

    const size_t material_id_for_cross[5][5] =
    {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    const size_t material_id_for_shell[6][6] =
    {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0}
    };

    FOR_O(1, 2)
    {
        const double g_yung_1 = 23.0;//1.0;// //31.23 * (1 - 1/(pow(0.25 * 7.0, 0.4)));
        const double g_puasson_1 = 
            // 0.21;
            0.20;
        const double g_yung_2 = 49.0;//10.0;//49.0; //
        const double g_puasson_2 = 
            // 0.21;
            0.28;
        // const double g_yung_2 = g_yung_1;
        // const double g_puasson_2 = g_puasson_1;

    time_t time_1 = time(NULL);

    {
        const double yung_1 = g_yung_1;
        const double puasson_1 = g_puasson_1;
        const double yung_2 = g_yung_2;
        const double puasson_2 = g_puasson_2;

        FILE *F_ch, *F_ch_fork;
        F_ch = fopen ("ch.gpd", "w");
        F_ch_fork = fopen ("ch_fork.gpd", "w");

        double i = 4;
        while (i < 132)
        {
            auto ch = ::get_ch(
                    yung_1, puasson_1, 
                    yung_2, puasson_2, (i*i)/(128.0*128.0));

            auto ch_fork = ::get_ch_fork(
                    yung_1, puasson_1, 
                    yung_2, puasson_2, (i*i)/(128.0*128.0));

            fprintf(F_ch, "%f %f %f %f %f %f %f %f %f %f %f %f\n", 
                    i*i,
                    ch[0],
                    ch[1],
                    ch[2],
                    ch[3],
                    ch[4],
                    ch[5],
                    ch[6],
                    ch[7],
                    ch[8],
                    ch[9],
                    ch[10]
                   );

            fprintf(F_ch_fork, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                    i*i,
                    ch_fork[0][0],
                    ch_fork[1][0],
                    ch_fork[2][0],
                    ch_fork[3][0],
                    ch_fork[4][0],
                    ch_fork[5][0],
                    ch_fork[6][0],
                    ch_fork[7][0],
                    ch_fork[8][0],
                    ch_fork[9][0],
                    ch_fork[10][0],
                    ch_fork[11][0],
                    ch_fork[0][1],
                    ch_fork[1][1],
                    ch_fork[2][1],
                    ch_fork[3][1],
                    ch_fork[4][1],
                    ch_fork[5][1],
                    ch_fork[6][1],
                    ch_fork[7][1],
                    ch_fork[8][1],
                    ch_fork[9][1],
                    ch_fork[10][1],
                    ch_fork[11][1]
                        );

            i += 4;
        };
        fclose(F_ch);
        fclose(F_ch_fork);
            auto ch_fork = ::get_ch_fork(
                    yung_1, puasson_1, 
                    yung_2, puasson_2, 0.0);
//            printf("%f\n", ch_fork[8][0]);
    };

    std::array<std::array<double, 3>, 3> matrix = 
    {{{1.0, 2.0, 4.0}, 
      {4.0, 9.0, 6.0},
      {7.0, 1.0, 8.0}}};

    double det = 
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

    printf("DDDDDDDDDD=%f %f\n", get_determinant<3>(matrix), det);
    printf("MMMMMMMMMM=%f %f\n", get_minor<3>(matrix, 2, 2), det);

    FOR_J(4, 5)
    {

    switch (j)
    {
        case 1:
            {
                FILE *F;
                F = fopen ("mata-quadrate.gpd", "w");

                {
        // const double yung_1 = g_yung_1;
        // const double puasson_1 = g_puasson_1;
        // const double yung_2 = g_yung_2;
        // const double puasson_2 = g_puasson_2;

                    // cdbl yung_2 = 40.0;
                    // cdbl yung_1 = 393.0;
                    // cdbl puasson_2 = 0.35;
                    // cdbl puasson_1 = 0.4;

                    cdbl yung_2 = 2.0;
                    cdbl yung_1 = 1.0;
                    cdbl puasson_2 = 0.25;
                    cdbl puasson_1 = 0.25;

                    dbl delta = 0.005;

                    double i = 0.98;//108
                    // while (i*i < 1.0)
                    {
                        glob_i = i;

                        dealii::Triangulation<2> tria;

                        const double dot[5] = 
                        {
                            (0.0),
                            // (delta/2.0),
                            // (0.5 - i / 2.0),
                            (0.25),
                            (0.5),
                            (0.75),
                            // (0.5 + i / 2.0),
                            // (1.0 - delta/2.0),
                            (1.0)
                        };

// 
                        // ::set_grid(tria);

                       ::set_tria <5> (tria, dot, material_id_for_quadrate);
//                        set_band<2> (tria, 64.0 - 128.0 / 6.0, 64.0 + 128.0 / 6.0, 0);
//                        set_band<2> (tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
                        //                ::set_quadrate<2>(tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
                        tria .refine_global (1);

                        {
                        std::ofstream out ("grid-igor.eps");
                        dealii::GridOut grid_out;
                        grid_out.write_eps (tria , out);
                        };
                        auto res = ::solved<2>(
                                tria, yung_1, puasson_1, yung_2, puasson_2); // /1.2
                        puts("dfd");

                        meta .push_back (res);

                        auto fork = ::get_fork <5> (
                                dot,material_id_for_quadrate, 
                                yung_1, puasson_1, yung_2, puasson_2);

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                                1.0 - i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1],
                                meta.back()[11]
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                1.0 - i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                              );

                        delta += 0.005;
                        i += 0.005;
                    };

                };

                fclose(F);
            };
            break;

        case 2:
            {
                FILE *F;
                F = fopen ("mata-cross.gpd", "w");

                {
                    double i = 4;
                    while (i < 96) //124.0
                    {
                        dealii::Triangulation<2> tria;

                        const double length = 96; //128.0 / 5.0 * 3.0;
                        const double width  = length - sqrt(length*length - i*i); // 128.0 / 5.0;

                        const double dot[6] = 
                        {
                            (0.0),
                            (64.0 - length / 2.0),
                            (64.0 - width  / 2.0),                    
                            (64.0 + width  / 2.0),                
                            (64.0 + length / 2.0),
                            (128.0)
                        };

                        ::set_tria <6> (tria, dot, material_id_for_cross);

        const double yung_1 = g_yung_1;
        const double puasson_1 = g_puasson_1;
        const double yung_2 = g_yung_2;
        const double puasson_2 = g_puasson_2;

                        std::ofstream out ("grid-cross.eps");
                        dealii::GridOut grid_out;
                        grid_out.write_eps (tria, out);
                        //                ::set_quadrate<2>(tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
                        tria .refine_global (3);

                        auto res = ::solved<2>(
                                tria, yung_1, puasson_1, yung_2, puasson_2); // /1.2

                        meta .push_back (res);

                        auto fork = ::get_fork <6> (
                                dot,material_id_for_cross, 
                                yung_1, puasson_1, yung_2, puasson_2);

                        auto ch = ::get_ch(
                                yung_1, puasson_1, 
                                yung_2, puasson_2, (i*i)/(128.0*128.0));

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1],
                                ch[0],
                                ch[1],
                                ch[2],
                                ch[3],
                                ch[4],
                                ch[5],
                                ch[6],
                                ch[7],
                                ch[8],
                                ch[9],
                                ch[10],
                                meta.back()[11]
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                              );

                        i += 4;
                    };
                };

                fclose(F);
            };
            break;

        case 3:
            {
                FILE *F;
                F = fopen ("mata-shell.gpd", "w");

                {
                    double i = 70;
                    while (i < 71.0)
                    {
                        dealii::Triangulation<2> tria;

                        const double length = 96.0; //128.0 / 5.0 * 3.0;
                        const double width  = (length - sqrt(length*length - i*i)) / 2.0; // 128.0 / 5.0;

                        const double dot[7] = 
                        {
                            (0.0),
                            (64.0 - length / 2.0),
                            (64.0 - length / 2.0 + width),
                            (64.0),
                            (64.0 + length / 2.0 - width),
                            (64.0 + length / 2.0),
                            (128.0)
                        };

                        ::set_tria <7> (tria, dot, material_id_for_shell);
                        //                ::set_quadrate<2>(tria, 64.0 - i / 2.0, 64.0 + i / 2.0, 0);
                        tria .refine_global (3);

    std::ofstream out ("grid-shell.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (tria, out);

        const double yung_1 = g_yung_1;
        const double puasson_1 = g_puasson_1;
        const double yung_2 = g_yung_2;
        const double puasson_2 = g_puasson_2;

                        auto res = ::solved<2>(         
                                tria, yung_1, puasson_1, yung_2, puasson_2); // /1.2

                        meta .push_back (res);

                        auto fork = ::get_fork <7> (
                                dot,material_id_for_shell,
                                yung_1, puasson_1, yung_2, puasson_2); // /1.2

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1],
                                meta.back()[11]
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                              );

                        i += 4;
                    };
                };

                fclose(F);
            };
            break;

        case 4:
            {
                FILE *F;
                F = fopen ("iso/mata-circ.gpd", "a");

                {
                    name_main_stress = 0;
                    double i = 0.30;
                    while (i < 0.31) //95.0//116.0
                    {
                        double x = i ;///sqrt(3.141592653589);//*sqrt(sqrt(3.0)/6.0)*1.28;

                        dealii::Triangulation<2> tria;

                        ::set_circ(tria, x, 5); // /sqrt(3.1415926535), 6);
                        // ::set_hexagon <60> (tria, 100.0, i * (sqrt(3.0) / 2.0));
                        // ::set_circ_in_hex(tria, 0.3, 6);
//                        tria .refine_global (3);

        const double yung_1 = g_yung_1;
        const double puasson_1 = g_puasson_1;
        const double yung_2 = g_yung_2;
        const double puasson_2 = g_puasson_2;

                        auto res = ::solved<2>(tria, 
                                yung_1, puasson_1, yung_2, puasson_2); 

                        meta .push_back (res);

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
//                                i*i, 
                                (x*x*3.14159265358979323846264338327),
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                meta.back()[11],
                                g_yung_1,
                                0.25 * 1.0
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                (x*x*3.141592653589),
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10]
                              );

                        i += 0.01;
                        name_main_stress += 1;
//                        break;
                    };
                };

                fclose(F);
            };
            break;

        case 5:
            {
                FILE *F;
                F = fopen ("mata-what.gpd", "w");

//                const size_t material[8][8] =
//                {
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                };

//                const size_t material[8][8] =
//                {
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 0, 0, 0, 0, 0, 0, 0},
//                };

//                const size_t material[8][8] =
//                {
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                    {0, 0, 1, 1, 1, 1, 0, 0},
//                };

//                const size_t material[8][8] =
//                {
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                    {0, 1, 1, 0, 0, 1, 1, 0},
//                };

//                const size_t material[8][8] =
//                {
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                    {0, 1, 0, 1, 0, 1, 0, 1},
//                };

//                const size_t material[8][8] =
//                {
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                    {1, 1, 1, 1, 0, 0, 0, 0},
//                };

                const size_t material[8][8] =
                {
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                    {1, 1, 0, 0, 1, 1, 0, 0},
                };

                {
                    double i = 80.0;
                    while (i < 84.0)
                    {
                        dealii::Triangulation<2> tria;

                        const double L = 1.0;
                        const double dot[9] = 
                        {
                            (0*L/8),
                            (1*L/8),
                            (2*L/8),
                            (3*L/8),
                            (4*L/8),
                            (5*L/8),
                            (6*L/8),
                            (7*L/8),
                            (8*L/8)
                        };

                        ::set_tria <9> (tria, dot, material);
                        tria .refine_global (2);

                        auto res = ::solved<2>(tria, 1.0, 0.20, 10.0, 0.28); // /1.2

                        meta .push_back (res);

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f\n", 
                                i*i*3.14159265358979323846264338327,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8]
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                i*i*3.141592653589793238462643383279,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8]
                              );

                        i += 4.0;
                    };
                };

                fclose(F);
            };
            break;

        case 6:
            {
                FILE *F;
                F = fopen ("mata-difficult.gpd", "w");

//                const size_t material[8][8] =
//                {
////                    {0, 0, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 1, 1, 1, 1, 1, 1, 0},
////                    {0, 0, 0, 0, 0, 0, 0, 0}
//                    {1, 0, 0, 1, 1, 0, 0, 1},
//                    {0, 0, 1, 0, 0, 1, 0, 0},
//                    {0, 1, 0, 0, 0, 0, 1, 0},
//                    {1, 0, 0, 1, 1, 0, 0, 1},
//                    {1, 0, 0, 1, 1, 0, 0, 1},
//                    {0, 1, 0, 0, 0, 0, 1, 0},
//                    {0, 0, 1, 0, 0, 1, 0, 0},
//                    {1, 0, 0, 1, 1, 0, 0, 1}
////                    {1, 1, 0, 1, 1, 0, 0, 1},
////                    {0, 1, 0, 0, 0, 0, 1, 0},
////                    {0, 0, 1, 0, 0, 1, 0, 0},
////                    {1, 0, 0, 1, 1, 0, 0, 1},
////                    {1, 0, 0, 1, 1, 0, 0, 1},
////                    {0, 0, 1, 0, 0, 1, 0, 0},
////                    {0, 1, 0, 0, 0, 0, 1, 0},
////                    {1, 0, 0, 1, 1, 0, 0, 1}
////                    {0, 1, 0, 0, 0, 0, 0, 0},
////                    {0, 0, 0, 0, 1, 0, 0, 0},
////                    {0, 1, 0, 1, 0, 0, 1, 0},
////                    {0, 0, 0, 0, 0, 0, 0, 0},
////                    {1, 0, 0, 0, 0, 1, 0, 0},
////                    {0, 0, 1, 0, 0, 0, 0, 0},
////                    {0, 0, 0, 0, 0, 1, 0, 1},
////                    {1, 0, 0, 1, 0, 0, 0, 0}
//                };
                const size_t material[13][13] =
                {
                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    {1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1},
                    {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1},
                    {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
                    {1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1},
                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
                    {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                    {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1},
                    {1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1},
                    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
//                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
//                    {0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
//                    {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
//                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
//                    {0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
//                    {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
//                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
//                    {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
//                    {0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
//                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
//                    {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
//                    {0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
//                    {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1}
//                    {1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
//                    {1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1},
//                    {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
//                    {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
//                    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
//                    {0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0},
//                    {1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1},
//                    {0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0},
//                    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
//                    {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
//                    {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
//                    {1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1},
//                    {1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}
                };

                {
                    double i = 60.0;
                    while (i < 64.0)
                    {
                        dealii::Triangulation<2> tria;

//                        const double dot[9] = 
//                        {
//                            (0.0),
//                            (1*16.0),
//                            (2*16.0),
//                            (3*16.0),
//                            (4*16.0),
//                            (5*16.0),
//                            (6*16.0),
//                            (7*16.0),
//                            (128.0)
//                        };
                        const double dot[14] = 
                        {
                            (0.0),
                            (1*128.0/13.0),
                            (2*128.0/13.0),
                            (3*128.0/13.0),
                            (4*128.0/13.0),
                            (5*128.0/13.0),
                            (6*128.0/13.0),
                            (7*128.0/13.0),
                            (8*128.0/13.0),
                            (9*128.0/13.0),
                            (10*128.0/13.0),
                            (11*128.0/13.0),
                            (12*128.0/13.0),
                            (128.0)
                        };

                        ::set_tria <14> (tria, dot, material);
                        tria .refine_global (2);

                        auto res = ::solved<2>(tria, 1.0, 0.25, 10.0, 0.25); // /1.2

                        meta .push_back (res);

                        auto fork = ::get_fork <14> (
                                dot,material, 1.0, 0.25, 10.0, 0.25);

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                               );

                        double mean = (fork[0] + fork[1]) / 2.0;
//                        double differ = 0.0;
//                        if (mean < meta.back()[10])

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                i*i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1],
                                mean,
                                (mean < meta.back()[10]) ? 
                                ((meta.back()[10] - mean) / mean) :
                                ((mean - meta.back()[10]) / meta.back()[10])
                              );

                        i += 4.0;
                    };
                };

                fclose(F);
            };
            break;

        case 7:
            {
                FILE *F;
                F = fopen ("mata-line.gpd", "w");

//                const size_t material[3][3] =
//                {
//                    {1, 1, 0, 0},
//                    {1, 1, 0, 0},
//                    {1, 1, 0, 0},
//                    {1, 1, 0, 0}
//                };
//                const size_t material[3][3] =
//                {
//                    {0, 1, 0},
//                    {0, 1, 0},
//                    {0, 1, 0}
//                };
                const size_t material[2][2] =
                {
                    {0, 1},
                    {0, 1}
                };

                {
                    double i = 0.45;
                    while (i < 0.46)
                    {
                        oMega = i;
                        dealii::Triangulation<2> tria;

//                        const double dot[4] = 
//                        {
//                            (0.0),
//                            (128.0/3.0),
//                            (2.0*128.0/3.0),
////                            (20.0),
////                            (i),
//                            (128.0)
////                            (0.0),
////                            (sqrt(0.2)/2),
////                            (sqrt(0.2)),
////                            (1.0)
//                        };

//                        const double dot[4] = 
//                        {
//                            (0.0),
//                            (64.0-40.0),
//                            (64.0+40.0),
//                            (128.0)
//                        };

                        const double dot[3] = 
                        {
                            (0.0),
                            (i),
                            (1.0)
                        };

                       ::set_tria <3> (tria, dot, material);
                        // ::set_line <10> (tria, 1.0, 1.0, i);
                       tria .refine_global (2);

                        auto res = ::solved<2>(tria, 1.0, 0.20, 10.0, 0.4); // /1.2

                        meta .push_back (res);

//                        auto fork = ::get_fork <4> (
//                                dot,material, 1.0, 0.25, 10.0, 0.25);
                        double fork[2] = {0.0};

                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
                                i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                               );

                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                                i,
                                meta.back()[0],
                                meta.back()[1],
                                meta.back()[2],
                                meta.back()[3],
                                meta.back()[4],
                                meta.back()[5],
                                meta.back()[6],
                                meta.back()[7],
                                meta.back()[8],
                                meta.back()[9],
                                meta.back()[10],
                                fork[0],
                                fork[1]
                              );

                        i += 0.05;
                    };
                };

                fclose(F);
            };
            break;

//        case 8:
//            {
//                FILE *F;
//                F = fopen ("mata-line.gpd", "w");
//
//                {
//                    const double mu1 = 4.0;
//                    const double mu2 = 0.4;//50.0/2.0;
//
//                    const double S = 0.5;
//
//                    double i = S;//sqrt(S);
//                    while (i < 1.01)
//                    {
//                        const double w = i;
//                        const double l = S / i;
//
//                        dealii::Triangulation<2> tria;
//
//                        ::set_rectangle(tria, 0.0, 0.0, l, w, 4);
//
//                        auto res = ::solved<2>(tria, 1.0, 0.25, 10.0, 0.25); // /1.2
//
//                        meta .push_back (res);
//
//                        double a1 = (mu1 * w + mu2 * (1.0 - w));
//                        double a2 = 1.0 / (l / mu1 + (1.0 - l) / mu2);
//
//                        double b1 = 1.0 / (l / a1  + (1.0 - l) / mu2);
//                        double b2 = (a2 * w + mu2 * (1.0 - w));
//
//                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
//                                i,
//                                meta.back()[0],
//                                meta.back()[1],
//                                meta.back()[2],
//                                meta.back()[3],
//                                meta.back()[4],
//                                meta.back()[5],
//                                meta.back()[6],
//                                meta.back()[7],
//                                meta.back()[8],
//                                meta.back()[9],
//                                meta.back()[10],
//                                b1,
//                                b2
//                               );
//
//                        printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
//                                i,
//                                meta.back()[0],
//                                meta.back()[1],
//                                meta.back()[2],
//                                meta.back()[3],
//                                meta.back()[4],
//                                meta.back()[5],
//                                meta.back()[6],
//                                meta.back()[7],
//                                meta.back()[8],
//                                meta.back()[9],
//                                meta.back()[10],
//                                b1,
//                                b2
//                              );
//                        break;
//                    };
//
//                fclose(F);
//            };
//            break;

        case 9: // Hexagon
            {
                FILE *F;
                F = fopen ("mata-hexagon.gpd", "w");

               // const size_t material[24][24] =
               // {
               //     {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}, 
               //     {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}, 
               //     {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, 
               //     {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, 
               //     {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}, 
               //     {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}, 
               //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 
               //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
               //     {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}, 
               //     {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}, 
               //     {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, 
               //     {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0}, 
               //     {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}, 
               //     {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}
               // };
               // const size_t material[24][24] =
               // {
               //     {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0}, 
               //     {0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0}, 
               //     {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0}, 
               //     {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
               //     {0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
               //     {0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0}, 
               //     {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
               //     {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0}, 
               //     {0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0}, 
               //     {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0}
               // };
//                const size_t material[24][24] =
//                {
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}
//                };
//                const size_t material[24][24] =
//                {
//                    {0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0}, 
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
//                    {0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1}, 
//                    {1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1}, 
//                    {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0}, 
//                    {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
//                    {0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0}
//                };
               // const size_t material[24][24] =
               // {
               //     {0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0}, 
               //     {0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0}, 
               //     {0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1}, 
               //     {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
               //     {1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1}, 
               //     {1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1}, 
               //     {1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1}, 
               //     {1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
               //     {0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0}, 
               //     {0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0}, 
               //     {0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0}
               // };

//                const size_t material[27][18] =
//                {
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0}, 
//                    {1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1}, 
//                    {0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1}, 
//                    {0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0}, 
//                    {0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0}, 
//                    {0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1}, 
//                    {0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
//                };
//    uint8_t width_2d_matrix = (dim + 1) * (dim + 1);
//    typename ElasticProblemSup<dim + 1>::TypeCoef coef;
//    coef[0][0][0][0][0] = 100.0;
//    coef[0][0][1][1][0] = 0.28;
//    coef[0][0][2][2][0] = 0.25;
//    coef[1][1][0][0][0] = 0.28;
//    coef[1][1][1][1][0] = 100.0;
//    coef[1][1][2][2][0] = 0.25;
//    coef[2][2][0][0][0] = 100.0;
//    coef[2][2][1][1][0] = 100.0;
//    coef[2][2][2][2][0] = 200.0;

                {
                    cdbl yung_2 = 40.0;
                    cdbl yung_1 = 393.0;
                    cdbl puasson_2 = 0.35;
                    cdbl puasson_1 = 0.4;

                    cdbl S   = 83.138438;
                    dbl share = 0.01;
                    // dbl i = 0.95;
                    dbl i = 0.0;
                    while (i*i < 0.999)
                    {
                        dealii::Triangulation<2> tria;

//                        ::set_hexagon <60> (tria, 100.0, i * (sqrt(3.0) / 2.0));
//                        ::set_hexagon_grid <60> (tria, 100.0, i);// / sqrt(2.0));
                        // ::set_hexagon_grid_pure (tria, 100.0, i);
                        ::set_hexagon_grid_pure (tria, 1.0, i);
//                        ::set_hexagon_brave <50> (tria, 100.0, 30.0);// / sqrt(2.0));
                        tria .refine_global (3);
//                        ::set_arbirary_tria (tria);

//                dealii::Point<2> center (0.0, 0.0);
//                dealii::Point<2> midle_p(2.0, -7.0);
//                printf("hex = %d\n", Hexagon(center, 10.0) .include(midle_p));

                        auto res = ::solved<2>(tria, yung_1, puasson_1, yung_2, puasson_2);
//
//                        double ch_V23 = 
//        (res[11]*res[8] - res[9]*res[8] - 4.0 * pow(res[6], 2.0) * res[9] * res[11]) /
//        (res[11]*res[8] + res[9]*res[8] + 4.0 * pow(res[6], 2.0) * res[9] * res[11]);
//
//                        printf("ch_V23=%f\n", ch_V23);
//
                       meta .push_back (res);
//
//                        double S = 6.0 * 100.0 * 100.0 * sqrt(3.0);
//                        fprintf(F, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
//                                4.0 * 3.14159265359 * pow((i * (sqrt(3.0) / (2.0))), 2.0) / S,
////                                3.0/2.0 * sqrt(3.0) * i * i / S,
////                                (6.0*sqrt(3.0)*i*i) / S,
//                                meta.back()[0],
//                                meta.back()[1],
//                                meta.back()[2],
//                                meta.back()[3],
//                                meta.back()[4],
//                                meta.back()[5],
//                                meta.back()[6],
//                                meta.back()[7],
//                                meta.back()[8],
//                                meta.back()[9],
//                                meta.back()[10],
//                                meta.back()[11],
//                                ch_V23
//                               );
//
//
                       fprintf(F,"%f %f %f %f %f %f %f %f %f %f %f %f\n", 
                               // 4.0 * 3.14159265359 * pow((i * (sqrt(3.0) / (2.0))), 2.0) / S,
                               1.0 - i*i,
                               meta.back()[0],
                               meta.back()[1],
                               meta.back()[2],
                               meta.back()[3],
                               meta.back()[4],
                               meta.back()[5],
                               meta.back()[6],
                               meta.back()[7],
                               meta.back()[8],
                               meta.back()[9],
                               meta.back()[10]
                             );

                       printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
                               // 4.0 * 3.14159265359 * pow((i * (sqrt(3.0) / (2.0))), 2.0) / S,
                               1.0 - i*i,
                               meta.back()[0],
                               meta.back()[1],
                               meta.back()[2],
                               meta.back()[3],
                               meta.back()[4],
                               meta.back()[5],
                               meta.back()[6],
                               meta.back()[7],
                               meta.back()[8],
                               meta.back()[9],
                               meta.back()[10]
                             );

                        // i += 0.005;
                        i += 0.05;
                    };
                };

                fclose(F);
            };
            break;
            };

};

    printf("%ld\n", (time(NULL) - time_1));
};

//    {
//        FILE *F;
//        F = fopen ("mata-cross.gpd", "w");
//
//        {
//            double i = 20;
//            while (i < 22)
//            {
//                dealii::Triangulation<2> tria;
//
//                const double dot[5] = 
//                {
//                    (0.0),
//                    (64.0 - i / 2.0),
//                    (64.0),
//                    (64.0 + i / 2.0),
//                    (128.0)
//                };
//
//                ::set_tria <5> (tria, dot, material_id_for_quadrate);
//                tria .refine_global (3);
//                auto res = ::solved<2>(tria, 100.0, 0.25, 1.0, 0.25); // /1.2
//
//                meta .push_back (res);
//
//                fprintf(F, "%f %f %f %f %f %f %f %f %f %f\n", 
//                        i*i*4.0,
//                        meta.back()[0],
//                        meta.back()[1],
//                        meta.back()[2],
//                        meta.back()[3],
//                        meta.back()[4],
//                        meta.back()[5],
//                        meta.back()[6],
//                        meta.back()[7],
//                        meta.back()[8]
//                       );
//
//                printf("\x1B[31m%f %f %f %f %f %f %f %f %f %f\x1B[0m\n", 
//                        i*i*4.0,
//                        meta.back()[0],
//                        meta.back()[1],
//                        meta.back()[2],
//                        meta.back()[3],
//                        meta.back()[4],
//                        meta.back()[5],
//                        meta.back()[6],
//                        meta.back()[7],
//                        meta.back()[8]
//                      );
//
//                i += 2;
//            };
//        };
//
//        fclose(F);
//    };

//    {
////        FILE *F;
////        F = fopen ("mata.gpd", "w");
//
//        {
//            size_t i = 52;
//            while (i < 54)
//            {
//                dealii::Triangulation<2> tria;
//
//                ::set_cross <2> (tria, 25.6, 76.8, 1);
//
//                auto res = ::solved<2>(tria, 100.0, 0.25, 1.0, 0.25);
//
//                meta .push_back (res);
//
////                fprintf (F, "%f %f %f\n", 4.0*i*i, res[0], res[1]);
//                printf ("%f %f %f\n", 4.0*i*i, res[0], res[1]);
//
//                i += 2;
//            };
//        };
//
////        fclose(F);
//    };

//    for(auto i : meta)
//        printf("%f %f\n", i[0], i[1]);

//        ::set_quadrate <2> (tria, 128.0 / 3.0, 2 * 128.0 / 3.0, 2);

//    {
//        size_t i = 2;
//        while (i < 62)
//        {
//            metas .push_back (::solved<2>(tria, 2.0, 0.25, 1.0, 0.25));
//            i += 2;
//        };
//    };
//    for (auto i : metas)
//        printf("%f %f\n", i[0], i[1]);

//    std::array<Femenist::Function<double, dim>, 3 > coef;

//    dealii::GridGenerator ::hyper_cube (tria, 0, 3);
//    std::vector< dealii::Point< 2 > > v (4);
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 4.0; v[1][1] = 0.0;
//    v[2][0] = 0.0; v[2][1] = 4.0;
//    v[3][0] = 4.0; v[3][1] = 4.0;
//
//    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;

//    dealii::Triangulation<2> tria;
//    set_hexagon_grid_pure(tria, 100.0, 50.0, 2);
//
//    std::vector<dealii::Point< 2 > > v (8);
//
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 1.0; v[1][1] = 0.0;
//    v[2][0] = 3.0; v[2][1] = 0.0;
//    v[3][0] = 4.0; v[3][1] = 0.0;
//    v[4][0] = 0.0; v[4][1] = 4.0;
//    v[5][0] = 1.0; v[5][1] = 4.0;
//    v[6][0] = 3.0; v[6][1] = 4.0;
//    v[7][0] = 4.0; v[7][1] = 4.0;
//
//    std::vector< dealii::CellData< 2 > > c (3, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 4;
//    c[0].vertices[3] = 5;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 5;
//    c[1].vertices[3] = 6;
//    c[1].material_id = 1;
//
//    c[2].vertices[0] = 2;
//    c[2].vertices[1] = 3;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    tria .create_triangulation (v, c, dealii::SubCellData());
//
//    tria .refine_global (1);
//
////    foo_2<2> (tria);
//    foo_3<2> (tria);
    dealii::Triangulation<2> tria;


//    int temp = v.length();
//    v.push_back(dealii::Point<2>(x0,y0));
//    v.push_back(dealii::Point<2>(x1,y1));
//    v.push_back(dealii::Point<2>(x0,y0));
//    v.push_back(dealii::Point<2>(x0,y0));
//    v.push_back(dealii::Point<2>(x0,y0));
//    v.push_back(dealii::Point<2>(x0,y0));
//    v.push_back(dealii::Point<2>(x6,y6));
//
//{
//    dealii::CellData<2> temp_c;
//
//    temp_c.vertices[0] = temp + 0;
//    temp_c.vertices[1] = 3;
//    temp_c.vertices[2] = 6;
//    temp_c.vertices[3] = 5;
//    temp_c.material_id = mat_id;
//
//    c.push_back(temp_c);
//}
//
//{
//    dealii::CellData<2> temp_c;
//
//    temp_c.vertices[0] = 1;
//    temp_c.vertices[1] = 4;
//    temp_c.vertices[2] = 6;
//    temp_c.vertices[3] = 3;
//    temp_c.material_id = mat_id;
//
//    c.push_back(temp_c);
//}
//
//{
//    dealii::CellData<2> temp_c;
//
//    temp_c.vertices[0] = 2;
//    temp_c.vertices[1] = 5;
//    temp_c.vertices[2] = 6;
//    temp_c.vertices[3] = 4;
//    temp_c.material_id = mat_id;
//
//    c.push_back(temp_c);
//}

//    std:vector<dealii::Point< 2 > > v (7);
//
//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = 1.0; v[1][1] = 0.0;
//    v[2][0] = 2.0; v[2][1] = 0.0;
//    v[3][0] = 0.5; v[3][1] = 1.0;
//    v[4][0] = 1.0; v[4][1] = 0.5;
//    v[5][0] = 1.5; v[5][1] = 1.0;
//    v[6][0] = 1.0; v[6][1] = 2.0;
//
//    std::vector< dealii::CellData< 2 > > c (3, dealii::CellData<2>());
//
//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 3;
//    c[0].vertices[3] = 4;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 4;
//    c[1].vertices[2] = 2;
//    c[1].vertices[3] = 5;
//    c[1].material_id = 1;
//
//    c[2].vertices[0] = 3;
//    c[2].vertices[1] = 6;
//    c[2].vertices[2] = 4;
//    c[2].vertices[3] = 5;
//    c[2].material_id = 0;

//    double radius = 50.0;
//
//    double Ro = 100.0;
//    double ro = radius;
//    double Ri = Ro * (sqrt(3.0) / 2.0);
//    double ri = ro * (sqrt(3.0) / 2.0);
//
//    double a[8] = {0.0, Ri-ri, Ri, 2.0*Ri, 3.0*Ri-ri, 3.0*Ri, 3.0*Ri+ri, 4.0*Ri};
//    double b[14] = {
//        0.0, ro/2.0, ro, Ro/2, Ro, 3.0*Ro/2.0 - ro, 3.0*Ro/2.0 - ro / 2.0,
//        3.0*Ro/2.0, 3.0*Ro/2.0 + ro / 2.0, 3.0*Ro/2.0 + ro 
//
//
    std::vector<dealii::Point< 2 > > v (4);

//    v[0][0] = 0.0; v[0][1] = 0.0;
//    v[1][0] = Ri-ri; v[1][1] = 0.0;
//    v[2][0] = Ri; v[2][1] = 0.0;
//    v[2][0] = Ri+ri; v[2][1] = 0.0;
//    v[2][0] = Ri+Ri; v[2][1] = 0.0;
//    v[3][0] = 0.0; v[3][1] = 1.0;
//    v[4][0] = 1.0; v[4][1] = 1.0;
//    v[5][0] = 2.0; v[5][1] = 1.0;

    v[0][0] = 0.0; v[0][1] = 0.0;
    v[1][0] = 1.0; v[1][1] = 0.0;
    v[2][0] = 1.0; v[2][1] = 1.0;
    v[3][0] = 0.0; v[3][1] = 1.0;
//    v[4][0] = 1.0; v[4][1] = 1.0;
//    v[5][0] = 2.0; v[5][1] = 1.0;
//    v[6][0] = 0.0; v[6][1] = 2.0;
//    v[7][0] = 1.0; v[7][1] = 2.0;
//    v[8][0] = 2.0; v[8][1] = 2.0;
//    v[4][0] = 1.0; v[4][1] = 2.0;
//    v[5][0] = 2.0; v[5][1] = 2.0;

    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());

    c[0].vertices[0] = 0;
    c[0].vertices[1] = 1;
    c[0].vertices[2] = 2;
    c[0].vertices[3] = 3;
    c[0].material_id = 0;

//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 5;
//    c[1].vertices[3] = 4;
//    c[1].material_id = 0;
//
//    c[2].vertices[0] = 6;
//    c[2].vertices[1] = 3;
//    c[2].vertices[2] = 4;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 8;
//    c[3].vertices[1] = 7;
//    c[3].vertices[2] = 4;
//    c[3].vertices[3] = 5;
//    c[3].material_id = 0;

//    c[1].vertices[0] = 1;
//    c[1].vertices[1] = 2;
//    c[1].vertices[2] = 5;
//    c[1].vertices[3] = 4;
//    c[1].material_id = 1;

//    printf("%d %d %d %d %d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3],
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]
//            );
//
//    dealii::GridReordering<2>::reorder_cells(c);
//
//    printf("%d %d %d %d %d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3],
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]
//            );

    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    dealii::GridReordering<2>::reorder_cells(c);
    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    tria .create_triangulation_compatibility (v, c, dealii::SubCellData());
//    tria .refine_global (2);

    std::ofstream out ("grid-1.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (tria, out);

    dealii::DoFHandler<2> dof(tria);
    dealii::FE_Q<2> fe(1);
    dof .distribute_dofs (fe);

    typename dealii::DoFHandler<2>::active_cell_iterator cell =
        dof.begin_active();

    // cell->child(0)->at_boundary();
    printf("TTTTTTTTTTTTTTTT%d\n", cell->line(0)->at_boundary());
    printf("TTTTTTTTTTTTTTTT%d\n", cell->at_boundary());
    for (st i = 0; i < 20; ++i)
    {
        printf("%f %f\n", i-10.0, std::atan(i-10.0) / 3.14159265359 * 180.0);
    };
    return 0;
}
////////////////////////////////////////////////////////////
