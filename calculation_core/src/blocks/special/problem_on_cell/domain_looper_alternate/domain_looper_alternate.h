#ifndef DOMAIN_LOOPER_ALTERNATE
#define DOMAIN_LOOPER_ALTERNATE 1

#include <deal.II/grid/grid_tools.h>

namespace OnCell
{
    template<st dim, st spec_dim>
        void loop_domain_alternate (const dealii::DoFHandler<dim> &dof_h,
                OnCell::BlackOnWhiteSubstituter &bows,
                dealii::DynamicSparsityPattern &dsp)
        {
            auto v_at_b = dealii::GridTools ::get_all_vertices_at_boundary (dof_h.get_tria());

            std::map<st, arr<st, spec_dim>> dof_at_b;
            {
                auto cell = dof_h.begin_active();
                auto endc = dof_h.end();
                for (; cell != endc; ++cell)
                {
                    for (st i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                    {
                        st v = cell -> vertex_index(i);
                        if (v_at_b.find(v) != v_at_b.end())
                        {
                            std::pair<st, arr<st, spec_dim>> dof;
                            dof.first = v;
                            for (st j = 0; j < spec_dim; ++j)
                            {
                                dof.second[j] = cell -> vertex_dof_index(i, j);
                            };
                            dof_at_b .insert (dof);
                        };
                    };
                };
            };

            arr<dbl, dim> max;
            for (st i = 0; i < dim; ++i)
            {
                max[i] = 0.0;
            };
            for (auto &&v : v_at_b)
            {
                for (st i = 0; i < dim; ++i)
                {
                    if (max[i] < v.second(i))
                        max[i] = v.second(i);
                };
            };

            ci8 direction[27][3] =
            {
                {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
                {-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
                {-1,  1, -1}, {0,  1, -1}, {1,  1, -1},
                {-1, -1,  0}, {0, -1,  0}, {1, -1,  0},
                {-1,  0,  0}, {0,  0,  0}, {1,  0,  0},
                {-1,  1,  0}, {0,  1,  0}, {1,  1,  0},
                {-1, -1,  1}, {0, -1,  1}, {1, -1,  1},
                {-1,  0,  1}, {0,  0,  1}, {1,  0,  1},
                {-1,  1,  1}, {0,  1,  1}, {1,  1,  1}
            };

            st max_num_black_points = 1;
            for (st i = 0; i < dim; ++i)
            {
                max_num_black_points *= 3;
            };

            for (auto&& white : v_at_b)
            {
                bool is_white = false;

                for (st i = 0; i < dim; ++i)
                {
                    if (white.second(i) < 1.0e-8)
                    {
                        is_white = true;
                        for (st j = 0; j < dim; ++j)
                        {
                            if (std::abs(white.second(j) - max[j]) < 1.0e-8)
                                is_white = false;
                        };
                        break;
                    };
                };

                if (is_white)
                {
                    for (st i = 0; i < max_num_black_points; ++i)
                    {
                        bool coor_suit = true;
                        arr<dbl, dim> coor_black;
                        for (st j = 0; j < dim; ++j)
                        {
                            coor_black[j] = white.second(j) + direction[i][j] * max[j];
                            if ((coor_black[j] < -1.0e-10) or (coor_black[j] > (max[j] + 1.0e-10)))
                                coor_suit = false;
                        };
                        if (coor_suit)
                        {
                            coor_suit = false;
                            for (st j = 0; j < spec_dim; ++j)
                            {
                                if (direction[i][j] != 0)
                                {
                                    coor_suit = true;
                                };
                            };
                        };
                        if (coor_suit)
                        {
                            for (auto&& black : v_at_b)
                            {
                                bool is_black = true;
                                for (st j = 0; j < dim; ++j)
                                {
                                    if (std::abs(black.second(j) - coor_black[j]) > 1.0e-8)
                                    {
                                        is_black = false;
                                        break;
                                    };
                                };
                                if (is_black)
                                {
                                    for (st j = 0; j < spec_dim; ++j)
                                    {
                                        // for (st k = 0; k < spec_dim; ++k)
                                        // {
                                            bows.add_white_and_black(
                                                    dof_at_b[white.first][j], 
                                                    dof_at_b[black.first][j]);
                                        // };
                                    };
                                };
                            };
                        };
                    };
                };
            };
            for (st i = 0; i < bows.size; ++i)
            {
                for (st j = 0; j < dof_h.n_dofs(); ++j)
                {
                    // if ((dsp.exists(bows.black[i], j)) and (not bows.is_black(j)))
                    if (dsp.exists(bows.black[i], j))
                    {
                        dsp.add(bows.white[i], bows.subst(j));
                        dsp.add(bows.subst(j), bows.white[i]);
                        // dsp.add(bows.white[i], j);
                        // dsp.add(j, bows.white[i]);
                    };
                    
                };
                // dsp.add(bows.white[i], bows.white[i]);
            };
        };
};

#endif
