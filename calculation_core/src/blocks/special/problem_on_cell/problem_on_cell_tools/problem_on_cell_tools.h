#ifndef problem_on_cell_tools_def
#define problem_on_cell_tools_def 1

#include "../../../../../../prmt_sintactic_addition/prmt_sintactic_addition.h"
#include <deal.II/lac/vector.h>

namespace OnCell
{
//! Массив для хранения метакоэффициентов, ячейковых функций и их производных, с доступом по векторному 
//  индексу в формате записи Горынина.
    template <typename T>
    class ArrayWithAccessToVector
    {
        public:
            ArrayWithAccessToVector () : size(0) {};

            ArrayWithAccessToVector (cst a_size) : size (a_size)//, empty(1)
            {
                content .resize (a_size);
                for (auto &&a : content)
                {
                    a .resize (a_size);
                    for (auto &&b : a)
                    {
                        b .resize (a_size);
                        // for (auto &&c : b)
                        // {
                        //     c = 0.0;
                        // };
                    };
                };
                // content[0][0][0] = 1.0; // Нулевое приближение ячейковой функции равно 1.0
                // for (st i = 0; i < size; ++i)
                // {
                //     for (st j = 0; j < size; ++j)
                //     {
                //         for (st k = 0; k < size; ++k)
                //         {
                //             content[i][j][k] = 0.0;
                //         };
                //     };
                // };
            };
            ArrayWithAccessToVector (ArrayWithAccessToVector &a) : size (a.size)//, empty(1)
            {
                content .resize (a.size);
                for (auto &&a : content)
                {
                    a .resize (a.size);
                    for (auto &&b : a)
                    {
                        b .resize (a.size);
                        // for (auto &&c : b)
                        // {
                        //     c = 0.0;
                        // };
                    };
                };
                for (st i = 0; i < a.size; ++i)
                {
                    for (st j = 0; j < a.size; ++j)
                    {
                        for (st k = 0; k < a.size; ++k)
                        {
                            content[i][j][k] = a.content[i][j][k];
                        };
                    };
                };
            };

            void reinit (cst a_size)
            {
                size = a_size;

                content .resize (a_size);
                for (auto &&a : content)
                {
                    a .resize (a_size);
                    for (auto &&b : a)
                    {
                        b .resize (a_size);
                    };
                };
            };

            ArrayWithAccessToVector<T>& operator= (const ArrayWithAccessToVector<T> &rhs)
            {
                if (size == rhs.size)
                {
                    for (st i = 0; i < size; ++i)
                    {
                        for (st j = 0; j < size; ++j)
                        {
                            for (st k = 0; k < size; ++k)
                            {
                                content[i][j][k] = rhs.content[i][j][k];
                            };
                        };
                    };
                }
                else
                {
                    printf("\x1B[31m ERROR in ArrayWithAccessToVector operator=, size != rhs.size \x1B[0m\n");
                };

                return *this;
            };


            static cst x = 0;
            static cst y = 1;
            static cst z = 2;

            T& operator [] (const arr<i32, 3> k)
            {
                // printf("%d %d %d, %ld\n", k[x], k[y], k[z], size);
                if ((k[x] >=   0) and (k[y] >=   0) and (k[z] >=   0) and
                    (k[x] < size) and (k[y] < size) and (k[z] < size))
                {
                    // puts("da");
                    return content[k[x]][k[y]][k[z]];
                } 
                else
                    return empty;
                // return content.at(k[x]).at(k[y]).at(k[z]);
            };

            const T& operator [] (const arr<i32, 3> k) const
            {
                // printf("%d %d %d, %ld\n", k[x], k[y], k[z], size);
                if ((k[x] >=   0) and (k[y] >=   0) and (k[z] >=   0) and
                    (k[x] < size) and (k[y] < size) and (k[z] < size))
                {
                    // puts("da");
                    return content[k[x]][k[y]][k[z]];
                } 
                else
                    return empty;
                // return content.at(k[x]).at(k[y]).at(k[z]);
            };

            vec<vec<vec<T>>> content;
            st size;
            T empty;
    };

};

#endif
