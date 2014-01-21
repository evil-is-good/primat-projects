#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

int main()
{
  CDT cdt;

  std::vector<Vertex_handle> vec_of_vertices;
  std::vector< std::vector<Vertex_handle> > vec_of_domains;
  

  double x_begin = -1.0;
  double x_end = 1.0;
  double step_by_x = 0.05;
  size_t num_node_x = (x_end - x_begin) / step_by_x + 1;
  double x_current;
  
  for(size_t item_node_x = 0; item_node_x < num_node_x; ++item_node_x)
  {
       x_current = x_begin + item_node_x * step_by_x;
       vec_of_vertices.push_back( cdt.insert( Point(x_current, std::pow(1.0 - x_current * x_current, 0.5) )) );  
  }

  for(size_t item_node_x = 1; item_node_x < num_node_x - 1; ++item_node_x)
  {
       x_current = x_end - item_node_x * step_by_x;
       vec_of_vertices.push_back( cdt.insert( Point(x_current, - std::pow(1.0 - x_current * x_current, 0.5) )) );  
  }

  
  vec_of_domains.push_back(vec_of_vertices);
  vec_of_vertices.clear();   
  

  vec_of_vertices.push_back(cdt.insert(Point(3.0, 3.0) ));
  vec_of_vertices.push_back(cdt.insert(Point(-3.0, 3.0) ));
  vec_of_vertices.push_back(cdt.insert(Point(-3.0, -3.0) ));
  vec_of_vertices.push_back(cdt.insert(Point(3.0, -3.0) ));

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

  list_of_seeds.push_back(Point(0.0, 0.0));


  std::cout << "Meshing the domain..." << std::endl;
  

  
  CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria(0.125, 0.5), true);
                                
  CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria(0.125, 0.5));


  std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
  std::cout << "Number of finite faces: " << cdt.number_of_faces() << std::endl;
  
 
  size_t mesh_faces_counter = 0;
  
  for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
      fit != cdt.finite_faces_end(); ++fit) 
  {
     if(fit->is_in_domain()) 
        ++mesh_faces_counter;

  }


  FILE *fp;
  if((fp = fopen("triangles.gpd", "w"))==NULL)
  {
     printf("He удается открыть файл.\n");
     exit(1);
  }
  
  FILE *fp1;
  if((fp1 = fopen("plot.gps", "w"))==NULL)
  {
     printf("He удается открыть файл.\n");
     exit(1);
  }

  CDT::Triangle trg;
  

   
  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
 	      trg = cdt.triangle(iter);
          fprintf( fp, "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", CGAL::to_double( trg[0].x() ),
                                                                    CGAL::to_double( trg[0].y() ),
                                                                    CGAL::to_double( trg[1].x() ),
                                                                    CGAL::to_double( trg[1].y() ),
                                                                    CGAL::to_double( trg[2].x() ),
                                                                    CGAL::to_double( trg[2].y() ));
  }

  fprintf( fp, "\n");
                                                                    
  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
 	      trg = cdt.triangle(iter);
          
          double middle_segment01_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()));
          double middle_segment01_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()));

          double middle_segment02_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[2].x()));
          double middle_segment02_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[2].y()));

          fprintf( fp, "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", middle_segment01_x,
                                                                    middle_segment01_y,
                                                                    middle_segment01_x,
                                                                    middle_segment01_y,
                                                                    middle_segment02_x,
                                                                    middle_segment02_y);
  }

  fprintf( fp, "\n");

  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
 	      trg = cdt.triangle(iter);
          
          double center_of_mass_x = (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[1].x()) +
                                     CGAL::to_double(trg[2].x())) / 3.0;

          double center_of_mass_y = (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[1].y()) +
                                     CGAL::to_double(trg[2].y())) / 3.0;

          fprintf( fp, "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", center_of_mass_x,
                                                                    center_of_mass_y,
                                                                    center_of_mass_x,
                                                                    center_of_mass_y,
                                                                    center_of_mass_x,
                                                                    center_of_mass_y);
  }


  fprintf( fp, "\n");
                                                                    
  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
 	      trg = cdt.triangle(iter);
          
          double middle_segment12_x = 0.5 * (CGAL::to_double(trg[1].x()) + CGAL::to_double(trg[2].x()));
          double middle_segment12_y = 0.5 * (CGAL::to_double(trg[1].y()) + CGAL::to_double(trg[2].y()));

          double middle_segment02_x = 0.5 * (CGAL::to_double(trg[0].x()) + CGAL::to_double(trg[2].x()));
          double middle_segment02_y = 0.5 * (CGAL::to_double(trg[0].y()) + CGAL::to_double(trg[2].y()));

          fprintf( fp, "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", middle_segment02_x,
                                                                    middle_segment02_y,
                                                                    middle_segment12_x,
                                                                    middle_segment12_y,
                                                                    middle_segment12_x,
                                                                    middle_segment12_y);
  }

  fprintf( fp, "\n");

  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
 	      trg = cdt.triangle(iter);
          fprintf( fp, "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f", CGAL::to_double( trg[0].x() ),
                                                                    CGAL::to_double( trg[0].y() ),
                                                                    CGAL::to_double( trg[1].x() ),
                                                                    CGAL::to_double( trg[1].y() ),
                                                                    CGAL::to_double( trg[2].x() ),
                                                                    CGAL::to_double( trg[2].y() ));
  }

  fprintf( fp, "\n");


    

  fprintf( fp1, "set term postscript color eps\n" 
                "set output \"./triang.eps\"\n" 
                "set nokey\n" 
                "set xlabel \"x\"\n"  
                "set ylabel \"y\"\n"
                "plot[-4:4][-4:4] \\\n" ); 
  
           
  
  int item_colum = 0;

  for(CDT::Finite_faces_iterator iter = cdt.finite_faces_begin(); iter != cdt.finite_faces_end(); ++iter)
  {
      if(iter->is_in_domain()) 
      {
           for(int i = 1; i < 6; i += 2)
           {
               fprintf( fp1, "                 \'./triangles.gpd\' u %d:%d with lines lt 1 lw 1 lc 1, \\\n", 
                        item_colum + i, item_colum + i + 1); 
           }
      }

      else
      {
           for(int i = 1; i < 6; i += 2)
           {
               fprintf( fp1, "                 \'./triangles.gpd\' u %d:%d with lines lt 1 lw 1 lc 3, \\\n", 
                        item_colum + i, item_colum + i + 1); 
           }
      }
      item_colum = item_colum + 6;
  }



  std::cout << "Number of faces in the mesh domain: " << mesh_faces_counter << std::endl;
  
}
