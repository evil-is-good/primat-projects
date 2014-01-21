#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <boost/foreach.hpp>

template <uint8_t dim>
struct Point
{
    double coor[dim];
    double value;
};

template <uint8_t dim>
void get_data (std::vector<Point<dim> > &points, 
        char* file_name)
{
    FILE *F;
    F = fopen (file_name, "r");
    {
        char ch[100];
        for (int i = 0; i < 7; ++i)
            fgets (ch, 100, F);
    };

    Point<dim> point;

    fscanf (F, "%lf %lf", &point.coor[0], &point.value);
    points .push_back (point);

    if (dim == 1)
    while (fscanf (F, "%lf %lf", &point.coor[0], &point.value) !=
            EOF)
    {
        if (fabs(point.coor[0] - points.back().coor[0]) > 1e-10)
            points .push_back (point);
    }
    else if (dim == 2)
    while (fscanf (F, "%lf %lf %lf", &point.coor[0], &point.coor[1],  &point.value) !=
            EOF)
    {
        if (fabs(point.coor[0] - points.back().coor[0]) > 1e-10)
            points .push_back (point);
    };


    fclose (F);

};

template <uint8_t dim>
void solved ()
{
    const uint8_t x = 0;
    const uint8_t y = 0;

    std::vector<Point<dim> > psi;
    std::vector<Point<dim> > T0;
    std::vector<Point<dim> > res;

    get_data (psi,
            "/home/primat/projects/tests/cell_heat_test/sources/res_1d.gpd");
    get_data (T0,
            "/home/primat/projects/tests/force_solve/sources/solution.gpd"); 

    double period = 2.0;
    double L = 4.0;
    double epsilon = period;// / L;

    res .resize (T0.size());
    res[0] = T0[0];
    res.back() = T0.back();

    for (int i = 1; i < T0.size() - 1; ++i)
    {
        for (int j = 1; j < dim; ++j)
            const double T0_deriv = 
                (T0[i + 1].value - T0[i - 1].value) / 
                (T0[i + 1].coor[j] - T0[i - 1].coor[j]);

        const double ksi = 
            ((T0[i].coor[x] / period) - 
             (size_t)(T0[i].coor[x] / period)) * period;
        double psi_value = 0.0;
        for (int j = 1; j < psi.size(); ++j)
        {
            if ((psi[j].coor[x] - ksi) > 1e-10)
            {
                printf("%f %f %f\n", psi[j].coor[x], ksi, psi[j].value);
                psi_value = 
                    ((ksi - psi[j - 1].coor[x]) / 
                    (psi[j].coor[x] - psi[j - 1].coor[x])) *
                    (psi[j].value - psi[j - 1].value) +
                    psi[j - 1].value;
                break;
            };
        };

        res[i].coor[x] = T0[i].coor[x];
        res[i].value = T0[i].value + psi_value * T0_deriv;// * epsilon;
        printf("%f %f %f %f %f\n", 
                res[i].value, T0[i].value, psi_value, T0_deriv, epsilon);
    };

    {
        FILE *F;
        F = fopen ("res.gpd", "w");
        for (Point<dim> p : res)
            fprintf(F, "%f %f\n", p.coor[x], p.value);
        fclose (F);
    };

    std::vector<Point<dim> > true_res;
    get_data (true_res,
            "/home/primat/projects/tests/force_solve/sources/true-res.gpd"); 
    {
        FILE *F;
        F = fopen ("dis.gpd", "w");
        for (int i = 0; i < res.size() ; ++i)
            fprintf(F, "%f %f\n", 
                    res[i].coor[x], 
                    fabs(res[i].value - true_res[i].value));
//                    (fabs(res[i].value - true_res[i].value) / true_res[i].value) * 100);
        fclose (F);
    };

};

int main(int argc, char *argv[])
{
    const uint8_t x = 0;
    const uint8_t dim = 1;
    std::vector<Point<dim> > psi;
    std::vector<Point<dim> > T0;
    std::vector<Point<dim> > res;
    get_data (psi,
            "/home/primat/projects/tests/cell_heat_test/sources/res_1d.gpd");
    get_data (T0,
            "/home/primat/projects/tests/force_solve/sources/solution.gpd"); 
    double period = 2.0;
    double L = 4.0;
    double epsilon = period;// / L;

    res .resize (T0.size());
    res[0] = T0[0];
    res.back() = T0.back();

    for (int i = 1; i < T0.size() - 1; ++i)
    {
        const double T0_deriv = 
            (T0[i + 1].value - T0[i - 1].value) / 
            (T0[i + 1].coor[x] - T0[i - 1].coor[x]);
        const double ksi = 
            ((T0[i].coor[x] / period) - 
             (size_t)(T0[i].coor[x] / period)) * period;
        double psi_value = 0.0;
        for (int j = 1; j < psi.size(); ++j)
        {
            if ((psi[j].coor[x] - ksi) > 1e-10)
            {
                printf("%f %f %f\n", psi[j].coor[x], ksi, psi[j].value);
                psi_value = 
                    ((ksi - psi[j - 1].coor[x]) / 
                    (psi[j].coor[x] - psi[j - 1].coor[x])) *
                    (psi[j].value - psi[j - 1].value) +
                    psi[j - 1].value;
                break;
            };
        };

        res[i].coor[x] = T0[i].coor[x];
        res[i].value = T0[i].value + psi_value * T0_deriv;// * epsilon;
        printf("%f %f %f %f %f\n", 
                res[i].value, T0[i].value, psi_value, T0_deriv, epsilon);
    };

    {
        FILE *F;
        F = fopen ("res.gpd", "w");
        for (Point<dim> p : res)
            fprintf(F, "%f %f\n", p.coor[x], p.value);
        fclose (F);
    };

    std::vector<Point<dim> > true_res;
    get_data (true_res,
            "/home/primat/projects/tests/force_solve/sources/true-res.gpd"); 
    {
        FILE *F;
        F = fopen ("dis.gpd", "w");
        for (int i = 0; i < res.size() ; ++i)
            fprintf(F, "%f %f\n", 
                    res[i].coor[x], 
                    fabs(res[i].value - true_res[i].value));
//                    (fabs(res[i].value - true_res[i].value) / true_res[i].value) * 100);
        fclose (F);
    };


    return 0;
}

