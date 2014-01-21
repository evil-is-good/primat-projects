template<uint8_t dim>
Report ElasticProblem2DOnCellV2<dim>::calculate_cells_stress ()
{
    cells_stress .clear ();

    dealii::QGauss<dim> quadrature_formula(2);

    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
            dealii::update_gradients | 
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const uint8_t dofs_per_cell = finite_element.dofs_per_cell;
    const uint8_t num_quad_points = quadrature_formula.size();

    typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        this->domain.dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        this->domain.dof_handler.end();

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    size_t cell_num = 0;
    for (; cell != endc; ++cell)
    {
            fe_values .reinit (cell);

            dbl area = 0.0;
            for(st i = 0; i < num_quad_points; ++i)
                area += fe_values.JxW(o);

            dbl mat_id = cell->material_id();

                
                std::array<dbl,2> deform;
                    for (auto i : {x, y}) 
                    {
                        deform[i][j] = 0.0;

                        for(auto i : {1, 2, 3, 4})
                        {
                    
                            dbl summ = 0.0;
                            for (size_t q_point = 0; q_point < num_quad_points; ++q_point)
                                summ += 
                                    fe_values.shape_grad (n, q_point)[i] *
                                    fe_values.JxW(q_point);
                            deform[i] += 
                                solution(cell->vertex_dof_index(n, 0)) * 
                                summ;
                        };
                        deform[i] /= area;
                    
                    };
    };
    REPORT_USE( 
            Report report;
            report.result = true;
            _return (report););
};


