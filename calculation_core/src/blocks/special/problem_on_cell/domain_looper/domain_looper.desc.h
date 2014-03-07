#ifndef DOMAIN_LOOPER_DESC_2

#define DOMAIN_LOOPER_DESC_2

#include <assert.h>

#include "projects/cae/main/black_on_white_substituter/black_on_white_substituter.h"
#include "projects/cae/main/loop_condition/loop_condition.h"

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/base/geometry_info.h>

namespace prmt
{
    template<u8 spacedim>
        class ToDealiiPoint
        {
            public:
                dealii::Point<spacedim> operator() (
                        const prmt::Point<spacedim> &p) const
                {
                    assert (false);
                    return dealii::Point<spacedim>();
                };
        };

    template<>
        class ToDealiiPoint<1>
        {
            public:
                dealii::Point<1> operator() (
                        const prmt::Point<1> &p) const
                {
                    return dealii::Point<1>(p.x());
                };
        };

    template<>
        class ToDealiiPoint<2>
        {
            public:
                dealii::Point<2> operator() (
                        const prmt::Point<2> &p) const
                {
                    return dealii::Point<2>(p.x(), p.y());
                };
        };

    template<>
        class ToDealiiPoint<3>
        {
            public:
                dealii::Point<3> operator() (
                        const prmt::Point<3> &p) const
                {
                    return dealii::Point<3>(p.x(), p.y(), p.z());
                };
        };

    template<u8 dim, u8 problemdim>
        class DomainLooper
        {
            enum {x, y, z};
            enum TypePoint {is_trivial, is_substitutive, is_substitutable};
            cdbl MIN_DISTANCE = 1e-10; // расстояние между "одинаковыми" точками
            struct LoopDoF
            {
                arr<st, problemdim> substitutiv;
                arr<vec<st>, problemdim> substitutable;
                vec<st> neighbor;
            };

            public:
            DomainLooper (const vec<prmt::LoopCondition<dim>> &loop_border);
            ~DomainLooper ();

            //METHODS
            public:
            Report loop_domain (const dealii::DoFHandler<dim> &dof_h,
                    prmt::BlackOnWhiteSubstituter &bows,
                    dealii::CompressedSparsityPattern &csp);
            private:
            ToDealiiPoint<dim> to_dp;

            void get_type_point_and_indxs (
                    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                    cst point_num,
                    TypePoint &type_point, st &substive_indx, st &substable_indx);

            void add_dofs_to_loop_dofs (
                    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                    cst point_num,
                    const TypePoint type_point, cst substive_indx, cst substable_indx);

            void if_point_black_or_white_add_to_loop_dofs (
                    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                    cst point_num);

            void set_ratio_black_to_white (BlackOnWhiteSubstituter &bows);

            void add_nodes_in_csp (const BlackOnWhiteSubstituter &bows,
                    dealii::CompressedSparsityPattern &csp);

            const vec<prmt::LoopCondition<dim>> loop_point;
            vec<LoopDoF> loop_dof;

            OPERATOR_REPORT;

            private:
            DomainLooper () {};

        };
};

#endif
