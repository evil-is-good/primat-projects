/*
 * =====================================================================================
 *
 *       Filename:  grid.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.07.2013 17:21:39
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
// #include "cgal/cgal.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <stdlib.h>
#include "projects/cae/main/point/point.h"

#include <stdint.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_reordering.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CDT::Vertex_handle Vertex_handle;
// typedef CDT::Point Point;


void convert_to_dealii_format(CDT &cdt, dealii::Triangulation<2> &triangulation)
{
    
    std::vector<dealii::Point<2> > vertex_in_grid;
    std::vector<dealii::CellData<2> > cell_in_grid;

    for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
    {
        CDT::Triangle trg = cdt.triangle(fit);

        // location of points on the triangle
        //
        //                    2
        //                   / \
        //                  /   \
        //                 /     \
        //                /       \
        //             5 /         \ 4
        //              /\         /\
        //             /  \       /  \
        //            /    \     /    \
        //           /      \   /      \
        //          /         6         \
        //         /          |          \
        //        /           |           \
        //       /            |            \
        //      0___________________________1
        //                    3
        //

        dealii::Point<2> p[7] =
        {
            dealii::Point<2>(trg[0].x(), trg[0].y()),
            dealii::Point<2>(trg[1].x(), trg[1].y()),
            dealii::Point<2>(trg[2].x(), trg[2].y()),

            (p[0] + p[1]) / 2.0,
            (p[1] + p[2]) / 2.0,
            (p[2] + p[0]) / 2.0,

            (p[0] + p[1] + p[2]) / 3.0
        };

        struct IndexPoint { u32 index; bool point_already_exist = false; };

        IndexPoint index_point[7];

        FOR(i, 0, 7)
            FOR(j, 0, vertex_in_grid.size())
                if (p[i].distance (vertex_in_grid[j]) < 1.e-10)
                {
                    index_point[i].index = j;
                    index_point[i].point_already_exist = true;
                };

        FOR(i, 0, 7)
            if (not index_point[i].point_already_exist)
            {
                vertex_in_grid .push_back (p[i]);
                index_point[i].index = vertex_in_grid.size() - 1;
                index_point[i].point_already_exist = true;
            };

        u8 mat_id = fit->is_in_domain() ? 0 : 1;

        auto add_cell = [&cell_in_grid, &index_point, mat_id] (arr<st, 4> &&indx) 
        {
            cell_in_grid .push_back (dealii::CellData<2>{
                    index_point[indx[0]].index,
                    index_point[indx[1]].index,
                    index_point[indx[2]].index,
                    index_point[indx[3]].index,
                    {.material_id = mat_id}});
        };

        add_cell (arr<st, 4>{0, 3, 6, 5});
        add_cell (arr<st, 4>{1, 4, 6, 3});
        add_cell (arr<st, 4>{2, 5, 6, 4});
    };

    dealii::GridReordering<2> ::reorder_cells (cell_in_grid);
    triangulation .create_triangulation_compatibility (
            vertex_in_grid, cell_in_grid, dealii::SubCellData());
};

enum class cguc_type
{
    Default
};
template <cguc_type type/*  = cguc_type::Default */>
void create_grid_using_cgal(CDT &cdt,
        vec<prmt::Point<2>> &outer_border,
        vec<prmt::Point<2>> &inner_border)
{};

template <>
void create_grid_using_cgal<cguc_type::Default> (CDT &cdt,
        vec<prmt::Point<2>> &outer_border,
        vec<prmt::Point<2>> &inner_border)
{
    std::vector<std::vector<Vertex_handle> > vec_of_domains;

    auto add_border_to_domain = [&vec_of_domains, &cdt] (vec<prmt::Point<2>> border)
    {
        std::vector<Vertex_handle> vec_of_vertices;
        for(auto p : border)
            vec_of_vertices.push_back(cdt.insert(CDT::Point(p.x(), p.y())));

        vec_of_domains.push_back(vec_of_vertices);
    };

    add_border_to_domain (inner_border);
    add_border_to_domain (outer_border);
    
    for(auto it = vec_of_domains.begin(); it != vec_of_domains.end(); ++it) 
    {
       for(auto vit = it->begin() + 1; vit != it->end(); ++vit)
       {
          cdt.insert_constraint( *(vit - 1), *vit );
       }
       cdt.insert_constraint( *( it->end() - 1), *( it->begin() ) );
    }
    
    std::list<CDT::Point> list_of_seeds;
    list_of_seeds.push_back(CDT::Point(0.5, 0.5));
    // std::list<Point> list_of_seeds_1;
    // list_of_seeds_1.push_back(Point(0.5, 0.5));
    // list_of_seeds_1.push_back(Point(0.2, 0.2));
    // std::list<Point> list_of_seeds_2;
    // list_of_seeds_2.push_back(Point(0.5, 0.5));

    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria(/* 0.125, 0.1 */), true);
    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria());
};

void set_grid(
        dealii::Triangulation< 2 > &triangulation,
        vec<prmt::Point<2>> outer_border,
        vec<prmt::Point<2>> inner_border)
{
    CDT cdt;
    create_grid_using_cgal<cguc_type::Default>(cdt, outer_border, inner_border);

    std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
    std::cout << "Number of finite faces: " << cdt.number_of_faces() << std::endl;
    
    size_t mesh_faces_counter = 0;
  
    for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) 
    {
        if(fit->is_in_domain()) 
            ++mesh_faces_counter;
    }
    
    std::cout << "Number of faces in the mesh domain: " << mesh_faces_counter << std::endl;
    
    convert_to_dealii_format(cdt, triangulation);
};

