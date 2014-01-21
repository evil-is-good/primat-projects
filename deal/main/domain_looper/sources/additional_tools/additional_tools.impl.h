#ifndef DOMAIN_LOOPER_ADDITIONAL_TOOLS_IMPL

#define DOMAIN_LOOPER_ADDITIONAL_TOOLS_IMPL

namespace DomainLoopsAdditionalTools
{
    template<uint8_t dim>
    Border<dim> get_borders (const dealii::DoFHandler<dim> &dof_h)
    {
        Border<dim> border;

        const uint8_t x = 0;
        const uint8_t y = 1;
        const uint8_t z = 2;
        const uint8_t wht = 0;
        const uint8_t blc = 1;

        border.coor[x][wht] = 0xffffffff;
        border.coor[y][wht] = 0xffffffff;
        border.coor[z][wht] = 0xffffffff;
        border.coor[x][blc] = -0xffffffff;
        border.coor[y][blc] = -0xffffffff;
        border.coor[z][blc] = -0xffffffff;

        
		for (
				typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_h.begin_active ();
				cell != dof_h.end ();
				++cell
			)
        {
            if (cell->at_boundary())
			    for(uint8_t i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                    for (uint8_t j = 0; j < dim; ++j)
                    {
                        if (cell->vertex(i)[j] < border.coor[j][0])
                            border.coor[j][0] = cell->vertex(i)[j];
                        if (cell->vertex(i)[j] > border.coor[j][1])
                            border.coor[j][1] = cell->vertex(i)[j];
                    };
        };

        std::cout << 1 << std::endl;
        for (uint8_t j = 0; j < dim; ++j)
            std::cout << border.coor[j][0] << " " << border.coor[j][1] << std::endl;
        std::cout << 2 << std::endl;

        return border;
    };

    template<uint8_t dim, bool type_space = 0>
    size_t num_black_or_white_dofs ()
    {
        return 1;
    };
    
    template<uint8_t dim, bool type_space = 0>
    Report get_list_black_and_white (
                const dealii::DoFHandler<dim> &dof_h,
                IndexAndCoor<dim, type_space> black_dofs,
                IndexAndCoor<dim, type_space> white_dofs)
    {
        Border<dim> border;
        border = get_borders<dim> (dof_h);
        REPORT_USE(
                Report report;
                report.result = false;
                report.ther_is_a_message = true;
                report.message = "dimention != (1 or 2 or 3)";
                _return(report);
                );
    };

    template<uint8_t dim, bool type_space = 0>
    Report get_blecks_neighbors (
                const dealii::DoFHandler<dim> &dof_h,
                const IndexAndCoor<dim, type_space> black_dofs,
                Neighbor<dim, type_space> neighbors)
    {
        REPORT_USE(
                Report report;
                report.result = false;
                report.ther_is_a_message = true;
                report.message = "dimention != (1 or 2 or 3)";
                _return(report);
                );
    };

    template<uint8_t dim, bool type_space = 0>
    Report get_ratio_black_to_white_and_modify_csp (
                const IndexAndCoor<dim, type_space> temp_black_dofs,
                const IndexAndCoor<dim, type_space> temp_white_dofs,
                Neighbor<dim, type_space> neighbors,
                size_t* white,
                size_t* black,
                dealii::CompressedSparsityPattern &csp)
    {
        REPORT_USE(
                Report report;
                report.result = false;
                report.ther_is_a_message = true;
                report.message = "dimention != (1 or 2 or 3)";
                _return(report);
                );
    };
};

#endif

#ifdef qwerty12345



    template<uint8_t dim, bool type_space = 0>
    Report get_list_black_and_white (
                const dealii::DoFHandler<dim> &dof_h,
                IndexAndCoor<dim, type_space>* temp_container_black_dofs,
                IndexAndCoor<dim, type_space>* temp_container_white_dofs,
                size_t*            black_angl_indexes,
                size_t             white_angl_index)
    {
        Border<dim> border; 

        const uint8_t x = 0;
        const uint8_t y = 1;
        const uint8_t z = 2;
        const uint8_t blc = 0;
        const uint8_t wht = 1;

        border = ::DomainLoopsAdditionalTools ::get_borders <dim> (dof_h);

        for (
				typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_h.begin_active ();
				cell != dof_h.end ();
				++cell
			)
        {
            if (cell->at_boundary())
			    for(uint8_t i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                {
                    uint8_t num_black = 0;
                    uint8_t num_white = 0;
                    for (uint8_t j = 0; j < dim; ++j)
                    {
                        if (cell->vertex(i)[j] == border[j][wht])
                            num_white++;

                        if (cell->vertex(i)[j] == border[j][blc])
                            num_black++;
                    };

                    if (num_white > 0)
                        if (num_white == dim)
                            for (uint8_t k = 0; k < type_space; ++k)
                                white_angl_index[k] = cell->get_dof_indeces(i)[k];
                        else
                            if ((num_white + num_black) == dim)
                                for (uint8_t k = 0; k < type_space; ++k)
                                {
                                    black_angl_index[n][k] = cell->get_dof_indeces(i)[k];
                                    n++;
                                };


        _return_report_true;
    };

 
    template<uint8_t dim, bool type_space = 0>
    Report get_blecks_neighbors (
                const dealii::DoFHandler<dim> &dof_h,
                const IndexAndCoor<dim, type_space>* temp_container_black_dofs,
                const size_t*            black_angl_indexes,
                std::vector<size_t>* a)
    {
        _return_report_true;
    };


    template<uint8_t dim, bool type_space = 0>
    Report get_ratio_black_to_white_and_modifi_csp  (
                const IndexAndCoor<dim, type_space>* temp_container_black_dofs,
                const IndexAndCoor<dim, type_space>* temp_container_white_dofs,
                const std::vector<size_t>* a,
                dealii::CompressedSparsityPattern &csp)
    {
        _return_report_true;
    };

#endif
