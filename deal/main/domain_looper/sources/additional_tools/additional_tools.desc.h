#ifndef DOMAIN_LOOPER_ADDITIONAL_TOOLS_DESC

#define DOMAIN_LOOPER_ADDITIONAL_TOOLS_DESC

#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include <stdint.h>

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/base/geometry_info.h>

namespace DomainLoopsAdditionalTools
{
    template<uint8_t dim>
    struct Border
    {
        double coor[dim][2]; // 0 - black, 1 - white;
    };

    template<uint8_t dim, bool type_space = 0>
    struct IndexAndCoor 
    {
        size_t index[dim][type_space + 1]; // dim - num sides
        double coor [dim];
    };

    template<uint8_t dim, bool type_space = 0>
    struct Neighbor 
    {
        std::vector<size_t> index[dim]; // dim - num sides
    };

    template<uint8_t dim>
    Border<dim> get_borders (const dealii::DoFHandler<dim> &dof_h);

    template<uint8_t dim, bool type_space = 0>
    size_t num_black_or_white_dofs ();
    
    template<uint8_t dim, bool type_space = 0>
    Report get_list_black_and_white (
                const dealii::DoFHandler<dim> &dof_h,
                IndexAndCoor<dim, type_space> black_dofs,
                IndexAndCoor<dim, type_space> white_dofs);

    template<uint8_t dim, bool type_space = 0>
    Report get_blecks_neighbors (
                const dealii::DoFHandler<dim> &dof_h,
                const IndexAndCoor<dim, type_space> black_dofs,
                Neighbor<dim, type_space> neighbors);

    template<uint8_t dim, bool type_space = 0>
    Report get_ratio_black_to_white_and_modify_csp (
                const IndexAndCoor<dim, type_space> temp_black_dofs,
                const IndexAndCoor<dim, type_space> temp_white_dofs,
                Neighbor<dim, type_space> neighbors,
                size_t* white,
                size_t* black,
                dealii::CompressedSparsityPattern &csp);
};

#endif
