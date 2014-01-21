/*
 * =====================================================================================
 *
 *       Filename:  head.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04.05.2013 12:47:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "../../elastic_test_on_cell/sources/cgal/cgal.h"

#include <stdlib.h>
#include <projects/deal/tests/heat_conduction_problem/heat_conduction_problem.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_reordering.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

void set_grid(dealii::Triangulation< 2 > &triangulation)
{
    std::vector<dealii::Point<2> > v;
    std::vector<dealii::CellData<2> > c;

    CDT cdt;
    std::vector<Vertex_handle> vec_of_vertices;
    std::vector<std::vector<Vertex_handle> > vec_of_domains;

//    double x_begin = -1.0;
//    double x_end = 1.0;
//    double dx = 0.05;
//    double x_current;
//    size_t num_node_x = (x_end - x_begin) / dx + 1;
//
//    double mult = 2.0;
//    double shift = 4.0;
//
//    for(size_t item_node_x = 0; item_node_x < num_node_x; ++item_node_x)
//    {
//         x_current = x_begin + item_node_x * dx;
//         vec_of_vertices.push_back( cdt.insert( 
//                     Point(x_current*mult+shift, 
//                         sqrt(1.0 - x_current * x_current)*mult+shift )) );  
//    }
//
//    for(size_t item_node_x = 1; item_node_x < num_node_x - 1; ++item_node_x)
//    {
//        x_current = x_end - item_node_x * dx;
//        vec_of_vertices.push_back( cdt.insert( 
//                    Point(x_current*mult+shift, 
//                        - sqrt(1.0 - x_current * x_current)*mult+shift )) );  
//    }
//    
    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0, 0.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0,  0.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(2.0/3.0,  1.0) ));
    vec_of_vertices.push_back(cdt.insert(Point(1.0/3.0,  1.0) ));

    vec_of_domains.push_back(vec_of_vertices);
    vec_of_vertices.clear();     
    
//    vec_of_vertices.push_back(cdt.insert(Point(0.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0, 0.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(128.0, 128.0) ));
//    vec_of_vertices.push_back(cdt.insert(Point(0.0, 128.0) ));
//
//    vec_of_domains.push_back(vec_of_vertices);
//    vec_of_vertices.clear();  
    
  double width_outer_domain = 1.0;
  double height_outer_domain = 1.0;
  
  double begin_x_outer_domain = 0.0;
  double begin_y_outer_domain = 0.0;
  
  size_t num_nodes_by_x = 0;
  size_t num_nodes_by_y = 0;
  
  double end_x_outer_domain = begin_x_outer_domain + width_outer_domain;
  double end_y_outer_domain = begin_y_outer_domain + height_outer_domain;
  

  
  double step_by_x = width_outer_domain / (num_nodes_by_x + 1); 
  double step_by_y = height_outer_domain / (num_nodes_by_y + 1); 
  
  for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain + item_node * step_by_x,
                                                 begin_y_outer_domain ) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain,
                                                 begin_y_outer_domain +  item_node * step_by_y) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_x; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( end_x_outer_domain - item_node * step_by_x,
                                                 end_y_outer_domain ) ));      
  }
  
  for(size_t item_node = 0; item_node <= num_nodes_by_y; ++item_node)
  {
     vec_of_vertices.push_back(cdt.insert(Point( begin_x_outer_domain,
                                                 end_y_outer_domain -  item_node * step_by_y) ));      
  }
  
    vec_of_domains.push_back(vec_of_vertices);
    vec_of_vertices.clear(); 
    
    for(auto it = vec_of_domains.begin(); it != vec_of_domains.end(); ++it) 
    {
       for(auto vit = it->begin() + 1; vit != it->end(); ++vit)
       {
          cdt.insert_constraint( *(vit - 1), *vit );
       }
       cdt.insert_constraint( *( it->end() - 1), *( it->begin() ) );
    }
    
    
    std::list<Point> list_of_seeds;
    list_of_seeds.push_back(Point(0.5, 0.5));

    std::cout << "Meshing the domain..." << std::endl;
  
    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria(), true);
    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria());
//    CGAL::refine_Delaunay_mesh_2(cdt, Criteria());

    std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
    std::cout << "Number of finite faces: " << cdt.number_of_faces() << std::endl;
    
    size_t mesh_faces_counter = 0;
  
    for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) 
    {
        if(fit->is_in_domain()) 
            ++mesh_faces_counter;
    }
    
    std::cout << "Number of faces in the mesh domain: " << mesh_faces_counter << std::endl;
    
    CDT::Triangle trg;
    
    size_t mat_id; 

    for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
    {
//        puts("!!!!!!!!!!!!!!");
        if(fit->is_in_domain())
        { 
            mat_id = 1;
        }
        else
        {
            mat_id = 0;
        }      
      
        trg = cdt.triangle(fit);
      
        size_t t = v.size()*10;
        size_t indx[7] = {t+0, t+1, t+2, t+3, t+4, t+5, t+6};
      
        double middle_segment01_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()));
        double middle_segment01_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()));
      
        double middle_segment12_x = 0.5 * (CGAL::to_double(trg[1].x()) + CGAL::to_double(trg[2].x()));
        double middle_segment12_y = 0.5 * (CGAL::to_double(trg[1].y()) + CGAL::to_double(trg[2].y()));

        double middle_segment02_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[2].x()));
        double middle_segment02_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[2].y()));
      
        double center_of_mass_x = (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()) +
                                   CGAL::to_double(trg[2].x())) / 3.0;
        double center_of_mass_y = (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()) +
                                   CGAL::to_double(trg[2].y())) / 3.0;
      
        FOR_I(0, v.size())
        {
            if (
                    (fabs(v[i](0) - trg[0].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[0].y()) < 1e-10))
                indx[0] = i;
            if (
                    (fabs(v[i](0) - trg[1].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[1].y()) < 1e-10))
                indx[1] = i;
            if (
                    (fabs(v[i](0) - trg[2].x()) < 1e-10) and
                    (fabs(v[i](1) - trg[2].y()) < 1e-10))
                indx[2] = i;
            if (
                    (fabs(v[i](0) - middle_segment01_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment01_y) < 1e-10))
                indx[3] = i;
            if (
                    (fabs(v[i](0) - middle_segment12_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment12_y) < 1e-10))
                indx[4] = i;
            if (
                    (fabs(v[i](0) - middle_segment02_x) < 1e-10) and
                    (fabs(v[i](1) - middle_segment02_y) < 1e-10))
                indx[5] = i;
            if (
                    (fabs(v[i](0) - center_of_mass_x) < 1e-10) and
                    (fabs(v[i](1) - center_of_mass_y) < 1e-10))
                indx[6] = i;
        };

        if (indx[0] == t)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
            indx[0] = v.size()-1;
        }
        if (indx[1] == t+1)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
            indx[1] = v.size()-1;
        }
        if (indx[2] == t+2)
        {
            v.push_back(dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
            indx[2] = v.size()-1;
        }
        if (indx[3] == t+3)
        {
            v.push_back(dealii::Point<2>( middle_segment01_x, middle_segment01_y ));
            indx[3] = v.size()-1;
        }
        if (indx[4] == t+4)
        {
            v.push_back(dealii::Point<2>( middle_segment12_x, middle_segment12_y ));
            indx[4] = v.size()-1;
        }
        if (indx[5] == t+5)
        {
            v.push_back(dealii::Point<2>( middle_segment02_x, middle_segment02_y ));
            indx[5] = v.size()-1;
        }
        if (indx[6] == t+6)
        {
            v.push_back(dealii::Point<2>( center_of_mass_x, center_of_mass_y));
            indx[6] = v.size()-1;
        }

//        printf("%d %d %d %d %d %d %d\n", 
//                indx[0],
//                indx[1],
//                indx[2],
//                indx[3],
//                indx[4],
//                indx[5],
//                indx[6]);

//        v.push_back(dealii::Point<2>(0.0, 0.0));
//        v.push_back(dealii::Point<2>(1.0, 0.0));
//        v.push_back(dealii::Point<2>(2.0, 0.0));
//        v.push_back(dealii::Point<2>(0.0, 2.0));
//        v.push_back(dealii::Point<2>(1.0, 2.0));
//        v.push_back(dealii::Point<2>(2.0, 2.0));

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[0];
            temp_c.vertices[1] = indx[3];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[5];
            temp_c.material_id = mat_id;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[1];
            temp_c.vertices[1] = indx[4];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[3];
            temp_c.material_id = mat_id;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[2];
            temp_c.vertices[1] = indx[5];
            temp_c.vertices[2] = indx[6];
            temp_c.vertices[3] = indx[4];
            temp_c.material_id = mat_id;

            c.push_back(temp_c);
        }


    }

    printf("%d %d\n", v.size(), c.size());
    puts("1");
//    dealii::GridReordering<2>::invert_all_cells_of_negative_grid(v, c);
    dealii::GridReordering<2>::reorder_cells(c);
    puts("2");
    triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
    puts("3");

};

