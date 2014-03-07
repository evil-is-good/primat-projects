#ifndef DOMAIN_LOOPER_IMPL_2

#define DOMAIN_LOOPER_IMPL_2

namespace prmt
{

    template<u8 dim, u8 problemdim>
        DomainLooper<dim, problemdim>::DomainLooper(
                const vec<prmt::LoopCondition<dim>> &loop_border) :
            loop_point(loop_border), loop_dof(loop_border.size())
    {
        FOR (i, 0, loop_border.size())
        {
            FOR (component, 0, problemdim)
                loop_dof[i].substitutable[component].resize(
                        loop_border[i].substitutable.size());
        };
    };

    template<uint8_t dim, u8 problemdim>
        void DomainLooper<dim, problemdim>::get_type_point_and_indxs (
                const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                cst point_num,
                TypePoint &type_point, st &substive_indx, st &substable_indx)
        {
            FOR (white_indx, 0, loop_point.size())
            {
                if (to_dp(loop_point[white_indx].substitutiv)
                        .distance (cell->vertex(point_num)) < MIN_DISTANCE)
                {
                    type_point    = is_substitutive;
                    substive_indx = white_indx;
                    break;
                };
                FOR (black_indx, 0, loop_point[white_indx].substitutable.size())
                {
                    if (to_dp(loop_point[white_indx].substitutable[black_indx])
                            .distance (cell->vertex(point_num)) < MIN_DISTANCE)
                    {
                        type_point     = is_substitutable;
                        substive_indx  = white_indx;
                        substable_indx = black_indx;
                        goto end_white_indx_loop;
                    };
                };
            };
            LABEL(end_white_indx_loop);
        };

    template<uint8_t dim, u8 problemdim>
        void DomainLooper<dim, problemdim>::add_dofs_to_loop_dofs (
                const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                cst point_num,
                const TypePoint type_point, cst substive_indx, cst substable_indx)
        {
            switch (type_point)
            {
                case is_substitutive:
                    {
            printf("white p1=(%f, %f), p2=(%f, %f), %ld %ld\n",
                    loop_point[substive_indx].substitutiv.x(),
                    loop_point[substive_indx].substitutiv.y(),
                    cell->vertex(point_num)(0),
                    cell->vertex(point_num)(1),
                    cell->vertex_dof_index (point_num, 0),
                    cell->vertex_dof_index (point_num, 1)
                    );
                        FOR (component, 0, problemdim)
                            loop_dof[substive_indx].substitutiv[component] = 
                            cell->vertex_dof_index (point_num, component);
                    };
                    break;
                case is_substitutable:
                    {
            printf("black p1=(%f, %f), p2=(%f, %f), %ld %ld  ",
                    loop_point[substive_indx].substitutable[substable_indx].x(),
                    loop_point[substive_indx].substitutable[substable_indx].y(),
                    cell->vertex(point_num)(0),
                    cell->vertex(point_num)(1),
                    cell->vertex_dof_index (point_num, 0),
                    cell->vertex_dof_index (point_num, 1)
                    );
                        FOR (component, 0, problemdim)
                            loop_dof[substive_indx]
                            .substitutable[component].at(substable_indx) =   
                            cell->vertex_dof_index (point_num, component);
            printf("%ld %ld %ld %ld\n",
                    loop_dof.at(substive_indx).substitutable[0].at(substable_indx),
                    loop_dof[substive_indx].substitutable[1].at(substable_indx),
                    substive_indx, substable_indx);

                        FOR (m, 0, dealii::GeometryInfo<dim>::vertices_per_cell)
                        {
                            if (point_num != m)
                                FOR (component, 0, problemdim)
                                    loop_dof[substive_indx].neighbor 
                                    .push_back (
                                            cell->vertex_dof_index (point_num, component));
                        };
                    };
                    break;
            };
        };

    template<uint8_t dim, u8 problemdim>
        void DomainLooper<dim, problemdim>::if_point_black_or_white_add_to_loop_dofs (
                const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                cst point_num)
    {
        TypePoint type_point = is_trivial;
        st substive_indx;
        st substable_indx;

        get_type_point_and_indxs (
                cell, point_num,
                and_assigned_to type_point, substive_indx, substable_indx);

        if (type_point != is_trivial)
            add_dofs_to_loop_dofs (
                    cell, point_num, type_point, substive_indx, substable_indx);
    };

    template<uint8_t dim, u8 problemdim>
        void DomainLooper<dim, problemdim>::set_ratio_black_to_white (
                BlackOnWhiteSubstituter &bows)
        {
            for (auto i : loop_dof)
                FOR (component, 0, problemdim)
                for (auto substable : i.substitutable[component]) 
                {
                    bows .add_white_and_black (
                            i.substitutiv[component], substable);
                };
        };

    template<uint8_t dim, u8 problemdim>
        void DomainLooper<dim, problemdim>::add_nodes_in_csp (
                const BlackOnWhiteSubstituter &bows,
                dealii::CompressedSparsityPattern &csp)
        {
            for (auto i : loop_dof)
                for (auto neighbor : i.neighbor)
                {
                    FOR (component, 0, problemdim)
                    {
                        csp .add (i.substitutiv[component], neighbor); 
                        csp .add (neighbor, i.substitutiv[component]); 
                    };
                };
        };

    template<uint8_t dim, u8 problemdim>
        Report DomainLooper<dim, problemdim>::loop_domain (
                const dealii::DoFHandler<dim> &dof_h, 
                BlackOnWhiteSubstituter &bows,
                dealii::CompressedSparsityPattern &csp)
        {

            if ((dof_h.n_boundary_dofs() > 0) && (not csp.empty()))
            {
                FILE *F;
                F = fopen("dof","w");
                for (
                        typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_h.begin_active ();
                        cell != dof_h.end ();
                        ++cell
                    )
                {
                    FOR (n, 0, dealii::GeometryInfo<dim>::vertices_per_cell)
                    {
                        if_point_black_or_white_add_to_loop_dofs (cell, n);
                    fprintf(F,"p=(%f, %f) i1=%d, i2=%d\n", 
                            cell->vertex(n)(0),
                            cell->vertex(n)(1),
                            cell->vertex_dof_index (n, 0),
                            cell->vertex_dof_index (n, 1));
                    };
                };
                fclose(F);

                // FOR (i, 0, loop_dof.size())
                // {
                //     printf("p and dof wp=(%f, %f) w1=%ld, w2=%ld ", 
                //             loop_point[i].substitutiv.x(),
                //             loop_point[i].substitutiv.y(),
                //             loop_dof[i].substitutiv[0],
                //             loop_dof[i].substitutiv[1]
                //           );
                //     FOR (component, 0, problemdim)
                //         FOR (j, 0, loop_dof[i].substitutable[component].size())
                //         {
                //             printf("bp=(%f, %f) b%ld=%ld ", 
                //                     loop_point[i].substitutable[j].x(),
                //                     loop_point[i].substitutable[j].y(),
                //                     component,
                //                     loop_dof[i].substitutable[component][j]);
                //         };
                //     printf("\n");
                // };

                set_ratio_black_to_white (bows);

                add_nodes_in_csp (bows, csp);

                // FOR(i, 0, dof_h.n_dofs())
                // FOR(j, 0, dof_h.n_dofs())
                // {
                //     csp .add (i, j); 
                //     csp .add (j, i); 
                // };
                // FOR(i, 0, bows.size)
                //     printf("%ld %ld\n", 
                //             bows.white[i],
                //             bows.black[i]);
                // printf("%d\n", loop_dof.size());
                // FOR(i, 0, loop_dof.size())
                // FOR (component, 0, problemdim)
                // FOR (j, 0, loop_dof[i].substitutable[component].size())
                //     printf("%ld %ld %d %d\n", 
                //             loop_dof[i].substitutiv[component], 
                //             loop_dof[i].substitutable[component][j],
                //             i, j);

                REPORT_USE(
                        Report report;
                        report.result = true;
                        _return(report);
                        );
            }
            else
            {
                REPORT_USE(
                        Report report;
                        report.result = false;
                        report.ther_is_a_message = true;
                        report.message = "";
                        if (dof_h.n_boundary_dofs() == 0)
                        report.message += " dof_h is empty";
                        if (csp.empty())
                        report.message += " csp is empty";
                        _return(report);
                        );
            };
        };

    template<uint8_t dim, u8 problemdim>
        DomainLooper<dim, problemdim>::~DomainLooper()
        {
        };

};

#endif

