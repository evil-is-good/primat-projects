/*
 * =====================================================================================
 *
 *       Filename:  solver_se.impl.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11.09.2012 13:12:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef FEMENIST_SOLVER_SE_IMPL

#define FEMENIST_SOLVER_SE_IMPL

#include <fstream> 

//#include "./solver_se.desc.h"

namespace Femenist
{
    template <class VECTOR>
    SolverSE<VECTOR>::SolverSE (dealii::SolverControl &cn)
        :
            solver_controller (cn)
    {

    };

    template <class VECTOR>
    template <class MATRIX>
    prmt::Report SolverSE<VECTOR>::solve(MATRIX &A,
                                         VECTOR &x,
                                         VECTOR &b)
    {
        std::vector<size_t> entry[x.size()];

        for (size_t i = 0; i < A.m(); ++i)
            for (size_t j = 0; j < A.n(); ++j)
                if (i != j)
                    if (A .el (i,j))
                        entry[i] .push_back (j);

//        for (size_t i = 0; i < A.m(); ++i)
//        {
//            for (size_t j = 0; j < A.n(); ++j)
//                if (A .el (i,j))
//                printf("A(%d,%d)=%f\n", i,j,A(i,j));
//            printf("\n");
//        };

//            for (size_t i = 0; i < x.size(); ++i)
//                printf("%ld %ld %f\n", i, entry[i].size(), b(i));
//                for (size_t j = 0; j < entry[i].size(); ++j)
//                    printf("%d %d %f\n", `<args>`);

    double cash[x.size()];
    for (size_t i = 0; i < x.size(); ++i)
        cash[i] = 100.0;

    size_t iteration = 0;
    double tolerance = 100.0;

    x(0) = 0.0; ////////////////////
    x(0) = 0.0; ////////////////////

    double relaxation_factor = 1.0;//1.9;

    std::ofstream ofs ("iterat");
    ofs.precision(12);
    double cash_2[x.size()];

    while (not solver_controller .check (iteration, tolerance))
    {
        tolerance = 0.0;

//        if (iteration == 1000) relaxation_factor = 3.0;

        ofs << (long double)x(12) << std::endl;


        for (size_t i = 0; i < x.size(); ++i)
        {
            cash_2[i] = cash[i];

            double difference = std::abs (cash[i] - x(i));
            //                printf("%f %f %f\n", cash[i], x(i),std::abs (cash[i] - x(i)) );

            if (difference > tolerance)
                tolerance = difference;

            cash[i] = x(i);
        };

        for (size_t i = 1; i < x.size(); ++i) //////////////////
        {
            x(i) = 0.0;//0.11458333333333333333;//0.0;

            for (size_t j = 0; j < entry[i].size(); ++j)
                x(i) -= A (i, entry[i][j]) * 
                    cash[entry[i][j]];
//                    x(entry[i][j]);

            x(i) += b(i);

            x(i) /= A (i, i);

//            if (iteration > 100)
//            {
//                double a = x(i) / 2.0 + cash_2[i] / 2.0 - cash[i];
//                double b = 2.0 * cash[i] - 3.0 * cash_2[i] / 2.0 - x(i) / 2.0;
//                double c = cash_2[i];
//
//                x(i) = a * 9.0 + b * 3.0 + c;
//            }
//            else
//            relaxation_factor = 1.9;
//            if ((iteration > 1000) and 
//                    (((fabs(cash_2[i] - cash[i]) / fabs(x(i))) < 1e-10) and
//                     ((fabs(cash[i] - x(i)) / fabs(x(i))) < 1e-10)))
//                relaxation_factor = 2.0;

            x(i) = cash[i] - relaxation_factor * (cash[i] - x(i));
        };

        ++iteration;
//        tolerance = 0.0;
    };

        //		  for (unsigned int i = 0; i<sizesol; ++i)
        //		  {
        //			  solution[item_sol](owerput.get_white_hole(i)) =
        //                  solution[item_sol](i);
        //		  }

        REPORT_USE(
                prmt::Report report;

                if (solver_controller .check (iteration, tolerance) _is 
                    dealii::SolverControl::success) then
                    report.result = true;
                else
                    report.result = false;

                report.ther_is_a_message = true;
                report.message = "Number of iterations: ";
                report.message += boost::lexical_cast<std::string>(iteration);
                report.message += ".  Max tolerance: ";
                report.message += boost::lexical_cast<std::string>(tolerance);

                _return(report);
                );
    };

};

#endif
