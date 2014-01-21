#ifndef DOMAIN_LOOPER_ADDITIONAL_TOOLS

#define DOMAIN_LOOPER_ADDITIONAL_TOOLS 

#include "./additional_tools.desc.h"
#include "./additional_tools.impl.h"

#include "./additional_tools.spec_dim1.desc.h"
#include "./additional_tools.spec_dim1.impl.h"

#include "./additional_tools.spec_dim2.desc.h"
#include "./additional_tools.spec_dim2.impl.h"

#include "./additional_tools.spec_dim3.desc.h"
#include "./additional_tools.spec_dim3.impl.h"

#endif
//        const uint8_t num_angles = dim * 2;
//        const uint8_t num_sides  = dim * 2;
//
//        const size_t num_boundary_dofs_per_side = 
//            (dof_h.n_boundary_dofs() - num_angles) / (num_sides);
//        std::vector<size_t> 
//            neighbors_for_bleck_dofs[num_boundary_dofs_per_side + 1];
//            // + 1, так как последними в списке ноходятся соседи для черных
//            // углов
//
//            num_boundary_dofs_per_side * (num_sides / 2) + (num_angles - 1);
//
        // делает тоже, что и следующая функция, но применительно к углам
//        {
//            for (uint8_t i = 0; i < (num_angles - 1); ++i)
//            {
//                black[i] = black_angl_indexes[i];
//                white[i] = white_angl_index;
//            };
//
//            uint8_t num_neighbors_for_bleck_angles =
//                neighbors_for_bleck_dofs[num_boundary_dofs_per_side] .size ();
//
//            uint8_t n = num_boundary_dofs_per_side; 
//            for (uint8_t i = 0; i < num_neighbors_for_bleck_angles; ++i)
//            {
//                csp .add (white_angl_index, neighbors_for_bleck_dofs[n][i]);
//                csp .add (neighbors_for_bleck_dofs[n][i], white_angl_index);
//            };
//        };
