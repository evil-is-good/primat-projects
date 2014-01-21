/*
 * =====================================================================================
 *
 *       Filename:  heat.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  17.09.2012 10:17:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "./head.h"

double const E  = 10;
double const Nu = 0.25;

template<int dim>
double boundary (const dealii::Point<dim> &p)
{
    return 1116.0*p(0);//0;//p(0);//(p(0)*p(0))*(p(1)*p(1));
};

template<int dim>
double source (const dealii::Point<dim> &p)
{
//    if ((p(0) > 2.0) and (p(0) < 6.0))
//        return -6.0*p(0);
//    else
//        return -12.0*p(0);

    return -2.0;//-6.0*p(0);//
    
    //-2 * p(0);//p(0) * (E - 2 * Nu);//-2.0*((p(0)*p(0)) + (p(1)*p(1)));
};

template <size_t num_points>
void set_tria(dealii::Triangulation< 2 > &triangulation, 
        const double points[num_points], 
        const size_t material_id[num_points - 1][num_points - 1])
{

    const size_t num_cells = num_points - 1;

    std::vector< dealii::Point< 2 > > v (num_points * num_points);

    FOR_I (0, num_points)
        FOR_J (0, num_points)
        {
            v[i * num_points + j] = dealii::Point< 2 >(points[j], points[i]);
        };

    std::vector< dealii::CellData< 2 > > c (
            num_cells * num_cells, dealii::CellData< 2 >());

    FOR_I (0, num_cells)
        FOR_J (0, num_cells)
        {
            c[i * num_cells + j].vertices[0] = i * num_points + j + 0;
            c[i * num_cells + j].vertices[1] = i * num_points + j + 1;
            c[i * num_cells + j].vertices[2] = i * num_points + j + num_points;
            c[i * num_cells + j].vertices[3] = i * num_points + j + num_points + 1;

            c[i * num_cells + j].material_id = material_id[i][j];
        };

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

void set_circ(dealii::Triangulation< 2 > &triangulation, 
        const double radius, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0, 128);

    triangulation.begin()->face(0) ->set_boundary_indicator (1);
    triangulation.begin()->face(1) ->set_boundary_indicator (1);

    triangulation .refine_global (4);
    {
        dealii::Point<2> center (64.0, 64.0);
        dealii::Triangulation<2>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            uint8_t count = 0;
            for (size_t i = 0; i < 4; ++i)
            {
                if (cell->vertex(i)(0) < ((64.0 + radius) + 1e-10) and
                        cell->vertex(i)(1) < ((64.0 + radius) + 1e-10) and   
                        cell->vertex(i)(0) > ((64.0 - radius) - 1e-10) and
                        cell->vertex(i)(1) > ((64.0 - radius) - 1e-10))   
                    count++;
                //                midle_p(0) += cell->vertex(i)(0);
                //                midle_p(1) += cell->vertex(i)(1);
            };
//            midle_p(0) /= 4;
//            midle_p(1) /= 4;

//            printf("%f %f\n", midle_p(0), midle_p(1));

//            if (center.distance(midle_p) < radius)
//            if (midle_p(0) < (64.0 + radius) and
//                midle_p(1) < (64.0 + radius) and   
//                midle_p(0) > (64.0 - radius) and
//                midle_p(1) > (64.0 - radius))   
            if (count >= 3)
            {
                cell->set_material_id(1);
            }
            else
                cell->set_material_id(0);
        };
    };
//    triangulation .refine_global (n_refine - 2);
};

void set_hexagon_grid_pure(dealii::Triangulation< 2 > &triangulation, 
        const double len_edge,
        const double radius)
{
    double Ro = len_edge;
    double ro = radius;
    double Ri = Ro * (sqrt(3.0) / 2.0);
    double ri = ro * (sqrt(3.0) / 2.0);

    printf("Radius %f %f\n", ro, ri);

    double a[7] = {0.0, Ri-ri, Ri, Ri+ri, 2.0*Ri, ri, 2.0*Ri - ri};
    double b[15] = {
        0.0, ro/2.0, ro, Ro/2, Ro, 1.5*Ro - ro, 1.5*Ro - ro / 2.0,
        1.5*Ro, 1.5*Ro + ro / 2.0, 1.5*Ro + ro, 2.0*Ro, 2.5*Ro, 3.0*Ro-ro,
        3.0*Ro-ro/2.0, 3.0*Ro};


    std::vector<dealii::Point< 2 > > v (30); //30

//    v[0][0]  = a[0]; v[0][1]  = b[0];
//    v[1][0]  = a[1]; v[1][1]  = b[0];
//    v[2][0]  = a[1]; v[2][1]  = b[1];
//    v[3][0]  = a[0]; v[3][1]  = b[1];
//    v[4][0]  = a[4]; v[4][1]  = b[3];
//    v[5][0]  = a[2]; v[5][1]  = b[4];
//    v[6][0]  = a[2]; v[6][1]  = b[10];
//    v[7][0]  = a[0]; v[7][1]  = b[11];
//    v[8][0]  = a[4]; v[8][1]  = b[11];
//    v[9][0]  = a[0]; v[9][1]  = b[14];
//    v[10][0] = a[2]; v[10][1] = b[14];
//    v[11][0] = a[4]; v[11][1] = b[14];
    

    v[0][0]  = a[0]; v[0][1]  = b[0];
    v[1][0]  = a[1]; v[1][1]  = b[0];
    v[2][0]  = a[2]; v[2][1]  = b[0];
    v[3][0]  = a[3]; v[3][1]  = b[0];
    v[4][0]  = a[4]; v[4][1]  = b[0];

    v[5][0]  = a[1]; v[5][1]  = b[1];
    v[6][0]  = a[3]; v[6][1]  = b[1];

    v[7][0]  = a[2]; v[7][1]  = b[2];

    v[8][0]  = a[0]; v[8][1]  = b[3];
    v[9][0]  = a[4]; v[9][1]  = b[3];

    v[10][0] = a[2]; v[10][1] = b[4];

    v[11][0] = a[0]; v[11][1] = b[5];
    v[12][0] = a[4]; v[12][1] = b[5];

    v[13][0] = a[5]; v[13][1] = b[6];
    v[14][0] = a[6]; v[14][1] = b[6];

    v[15][0] = a[5]; v[15][1] = b[8];
    v[16][0] = a[6]; v[16][1] = b[8];

    v[17][0] = a[0]; v[17][1] = b[9];
    v[18][0] = a[4]; v[18][1] = b[9];

    v[19][0] = a[2]; v[19][1] = b[10];

    v[20][0] = a[0]; v[20][1] = b[11];
    v[21][0] = a[4]; v[21][1] = b[11];

//    v[23][0] = a[4]; v[23][1] = b[8];

    v[22][0] = a[2]; v[22][1] = b[12];

    v[23][0] = a[1]; v[23][1] = b[13];
    v[24][0] = a[3]; v[24][1] = b[13];

    v[25][0] = a[0]; v[25][1] = b[14];
    v[26][0] = a[1]; v[26][1] = b[14];
    v[27][0] = a[2]; v[27][1] = b[14];
    v[28][0] = a[3]; v[28][1] = b[14];
    v[29][0] = a[4]; v[29][1] = b[14];
//
////    v[31][0] = a[4]; v[31][1] = b[8];  // 13

    std::vector< dealii::CellData< 2 > > c (20, dealii::CellData<2>()); //20

//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 5;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;
//
//    c[1].vertices[0] = 5;
//    c[1].vertices[1] = 1;
//    c[1].vertices[2] = 2;
//    c[1].vertices[3] = 4;
//    c[1].material_id = 0;
//
//    c[2].vertices[0] = 3;
//    c[2].vertices[1] = 5;
//    c[2].vertices[2] = 6;
//    c[2].vertices[3] = 7;
//    c[2].material_id = 0;
//
//    c[3].vertices[0] = 5;
//    c[3].vertices[1] = 4;
//    c[3].vertices[2] = 8;
//    c[3].vertices[3] = 6;
//    c[3].material_id = 0;
//
//    c[4].vertices[0] = 7;
//    c[4].vertices[1] = 6;
//    c[4].vertices[2] = 10;
//    c[4].vertices[3] = 9;
//    c[4].material_id = 0;
//
//    c[5].vertices[0] = 6;
//    c[5].vertices[1] = 8;
//    c[5].vertices[2] = 11;
//    c[5].vertices[3] = 10;
//    c[5].material_id = 0;

//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);
//
////    dealii::GridReordering<2,2>::invert_all_cells_of_negative_grid 
////        (v, c);
//    dealii::GridReordering<2>::reorder_cells(c);
//
//    puts("/////////////////////////////////");
//
//    printf("%d %d %d %d\n",
//    c[0].vertices[0],
//    c[0].vertices[1],
//    c[0].vertices[2],
//    c[0].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[1].vertices[0],
//    c[1].vertices[1],
//    c[1].vertices[2],
//    c[1].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[2].vertices[0],
//    c[2].vertices[1],
//    c[2].vertices[2],
//    c[2].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[3].vertices[0],
//    c[3].vertices[1],
//    c[3].vertices[2],
//    c[3].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[4].vertices[0],
//    c[4].vertices[1],
//    c[4].vertices[2],
//    c[4].vertices[3]);
//
//    printf("%d %d %d %d\n",
//    c[5].vertices[0],
//    c[5].vertices[1],
//    c[5].vertices[2],
//    c[5].vertices[3]);



//    c[0].vertices[0] = 0;
//    c[0].vertices[1] = 1;
//    c[0].vertices[2] = 2;
//    c[0].vertices[3] = 3;
//    c[0].material_id = 0;

    c[0].vertices[0] = 1;
    c[0].vertices[1] = 5;
    c[0].vertices[2] = 8;
    c[0].vertices[3] = 0;
    c[0].material_id = 0;

    c[1].vertices[0] = 1;
    c[1].vertices[1] = 2;
    c[1].vertices[2] = 7;
    c[1].vertices[3] = 5;
    c[1].material_id = 1;

    c[2].vertices[0] = 2;
    c[2].vertices[1] = 3;
    c[2].vertices[2] = 6;
    c[2].vertices[3] = 7;
    c[2].material_id = 1;

    c[3].vertices[0] = 3;
    c[3].vertices[1] = 4;
    c[3].vertices[2] = 9;
    c[3].vertices[3] = 6;
    c[3].material_id = 0;

    c[4].vertices[0] = 8;
    c[4].vertices[1] = 5;
    c[4].vertices[2] = 7;
    c[4].vertices[3] = 10;
    c[4].material_id = 0;

    c[5].vertices[0] = 7;
    c[5].vertices[1] = 6;
    c[5].vertices[2] = 9;
    c[5].vertices[3] = 10;
    c[5].material_id = 0;

    c[6].vertices[0] = 8;
    c[6].vertices[1] = 10;
    c[6].vertices[2] = 13;
    c[6].vertices[3] = 11;
    c[6].material_id = 0;

    c[7].vertices[0] = 10;
    c[7].vertices[1] = 9;
    c[7].vertices[2] = 12;
    c[7].vertices[3] = 14;
    c[7].material_id = 0;

    c[8].vertices[0] = 11;
    c[8].vertices[1] = 13;
    c[8].vertices[2] = 15;
    c[8].vertices[3] = 17;
    c[8].material_id = 1;

    c[9].vertices[0] = 13;
    c[9].vertices[1] = 10;
    c[9].vertices[2] = 19;
    c[9].vertices[3] = 15;
    c[9].material_id = 0;

    c[10].vertices[0] = 10;
    c[10].vertices[1] = 14;
    c[10].vertices[2] = 16;
    c[10].vertices[3] = 19;
    c[10].material_id = 0;

    c[11].vertices[0] = 14;
    c[11].vertices[1] = 12;
    c[11].vertices[2] = 18;
    c[11].vertices[3] = 16;
    c[11].material_id = 1;

    c[12].vertices[0] = 17;
    c[12].vertices[1] = 15;
    c[12].vertices[2] = 19;
    c[12].vertices[3] = 20;
    c[12].material_id = 0;

//    c[13].vertices[0] = 15;
//    c[13].vertices[1] = 17;
//    c[13].vertices[2] = 31; //// 31
//    c[13].vertices[3] = 19;
//    c[13].material_id = 1;

    c[13].vertices[0] = 16;
    c[13].vertices[1] = 18;
    c[13].vertices[2] = 21;
    c[13].vertices[3] = 19;
    c[13].material_id = 0;

    c[14].vertices[0] = 20;
    c[14].vertices[1] = 19;
    c[14].vertices[2] = 22;
    c[14].vertices[3] = 23;
    c[14].material_id = 0;

    c[15].vertices[0] = 19;
    c[15].vertices[1] = 21;
    c[15].vertices[2] = 24;
    c[15].vertices[3] = 22;
    c[15].material_id = 0;

    c[16].vertices[0] = 20;
    c[16].vertices[1] = 23;
    c[16].vertices[2] = 26;
    c[16].vertices[3] = 25;
    c[16].material_id = 0;

    c[17].vertices[0] = 23;
    c[17].vertices[1] = 22;
    c[17].vertices[2] = 27;
    c[17].vertices[3] = 26;
    c[17].material_id = 1;

    c[18].vertices[0] = 22;
    c[18].vertices[1] = 24;
    c[18].vertices[2] = 28;
    c[18].vertices[3] = 27;
    c[18].material_id = 1;

    c[19].vertices[0] = 24;
    c[19].vertices[1] = 21;
    c[19].vertices[2] = 29;
    c[19].vertices[3] = 28;
    c[19].material_id = 0;

    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    dealii::GridReordering<2>::reorder_cells(c);
    printf("%d %d %d %d\n",
    c[0].vertices[0],
    c[0].vertices[1],
    c[0].vertices[2],
    c[0].vertices[3]
    );
    triangulation .create_triangulation_compatibility (v, c, dealii::SubCellData());
//    triangulation .refine_global (n_ref);

    std::ofstream out ("grid-2.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (triangulation , out);
};

void set_grid_alternate(dealii::Triangulation< 2 > &triangulation)
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

    size_t count = 0;

//    for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
//    {
////        puts("!!!!!!!!!!!!!!");
//        if(fit->is_in_domain())
//        { 
//            mat_id = 1;
//        }
//        else
//        {
//            mat_id = 0;
//        }      
//      
//        trg = cdt.triangle(fit);
//      
//        bool   flag[3] = {true, true, true};
//        size_t indx[4] = {0};
//      
//        FOR_I(0, v.size())
//        {
////            bool brk = false;
//
//            FOR_J(0, 3)
//                if (flag[j])
//            {
////                printf("%f %f %f %f\n", v[i]);
//                if (
//                        (fabs(v[i](0) - trg[j].x()) < 1e-10) and
//                        (fabs(v[i](1) - trg[j].y()) < 1e-10))
//                {
//                    indx[j] = i;
//                    flag[j] = false;
////                    brk = true;
//                    printf("vv %f %f\n", v[i](0), v[i](1));
////                    break;
//                };
//            };
//
////            if (brk) break;
//        };
//
//        FOR_I(0, 3)
//        {
//            if (flag[i])
//            {
//                v .push_back (
//                        dealii::Point<2> (
//                            CGAL::to_double(trg[i].x()),  CGAL::to_double(trg[i].y())));
//                indx[i] = v.size() - 1;
//            };
//        };
//        printf("%d %d %d %d %d %d\n", flag[0], flag[1], flag[2], indx[0], indx[1], indx[2]);
//
//        if (flag[2])
//        {
//            v .push_back (
//                    dealii::Point<2> (
//                        CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y())));
//            indx[3] = v.size() - 1;
//        }
//        else
//            indx[3] = indx[2] + 1;
//        printf("%d %d %d %d %d %d %d\n", flag[0], flag[1], flag[2], indx[0], indx[1], indx[2], indx[3]);
//        
////        printf("%d %d %d %d %d %d %d\n", 
////                indx[0],
////                indx[1],
////                indx[2],
////                indx[3],
////                indx[4],
////                indx[5],
////                indx[6]);
//
//        dealii::CellData<2> temp_c;
//
//        temp_c.vertices[0] = indx[0];
//        temp_c.vertices[1] = indx[1];
//        temp_c.vertices[2] = indx[2];
//        temp_c.vertices[3] = indx[3];
//        temp_c.material_id = mat_id;
//
//        c.push_back(temp_c);
//
//        FOR_I(0, 3)
//            printf("%f %f\n", CGAL::to_double(trg[i].x()),  CGAL::to_double(trg[i].y()));
//        printf("\n");
//
//        FOR_I(0, 4)
//            printf("%f %f %d\n", v[indx[i]](0), v[indx[i]](1), flag[i % 3]);
//        printf("\n");
//
//        ++count;// printf("count %d\n", count);
//
//        if (count == 7)
//            break;
//    };

//    FOR_I(0, v.size())
//        printf("%f %f\n", v[i](0), v[i](1));

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
            v.push_back(
                    dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
            indx[0] = v.size()-1;
        }
        if (indx[1] == t+1)
        {
            v.push_back(
                    dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
            indx[1] = v.size()-1;
        }
        if (indx[2] == t+2)
        {
            v.push_back(
                    dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
            indx[2] = v.size()-1;
        }
        if (indx[3] == t+3)
        {
            v.push_back(
                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
//                    dealii::Point<2>( CGAL::to_double(trg[1].x()),  CGAL::to_double(trg[1].y()) ) );
//                    dealii::Point<2>( middle_segment01_x, middle_segment01_y ));
            indx[3] = v.size()-1;
        }
        if (indx[4] == t+4)
        {
            v.push_back(
                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
//                    dealii::Point<2>( CGAL::to_double(trg[2].x()),  CGAL::to_double(trg[2].y()) ) );
//                    dealii::Point<2>( middle_segment12_x, middle_segment12_y ));
            indx[4] = v.size()-1;
        }
        if (indx[5] == t+5)
        {
            v.push_back(
                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
//                    dealii::Point<2>( CGAL::to_double(trg[0].x()),  CGAL::to_double(trg[0].y()) ) );
//                    dealii::Point<2>( middle_segment02_x, middle_segment02_y ));
            indx[5] = v.size()-1;
        }
        if (indx[6] == t+6)
        {
            v.push_back(
                    dealii::Point<2>( center_of_mass_x, center_of_mass_y));
            indx[6] = v.size()-1;
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[3];
            temp_c.vertices[1] = indx[0];
            temp_c.vertices[2] = indx[1];
            temp_c.vertices[3] = indx[6];
            temp_c.material_id = mat_id;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[4];
            temp_c.vertices[1] = indx[1];
            temp_c.vertices[2] = indx[2];
            temp_c.vertices[3] = indx[6];
            temp_c.material_id = mat_id;

            c.push_back(temp_c);
        }

        {
            dealii::CellData<2> temp_c;

            temp_c.vertices[0] = indx[5];
            temp_c.vertices[1] = indx[2];
            temp_c.vertices[2] = indx[0];
            temp_c.vertices[3] = indx[6];
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

void set_single(dealii::Triangulation< 2 > &triangulation)
{
    std::vector< dealii::Point< 2 > > v (4);

    v[0][0] = 0.0; v[0][1] = 0.0;
    v[1][0] = 0.0; v[1][1] = 1.0;
    v[2][0] = 1.0; v[2][1] = 0.0;
    v[3][0] = 1.0; v[3][1] = 0.0;

    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());

    c[0].vertices[0] = 0;
    c[0].vertices[1] = 1;
    c[0].vertices[2] = 2;
    c[0].vertices[3] = 3;
    c[0].material_id = 0;

    triangulation .create_triangulation (v, c, dealii::SubCellData());
};

template<uint8_t dim>
dealii::Point<dim, double> get_grad (
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        const dealii::Vector<double> solution,
        uint8_t index_vertex)
{
    dealii::Point<dim, double> grad;

    double x1 = cell->vertex(0)(0);
    double x2 = cell->vertex(1)(0);
    double x3 = cell->vertex(2)(0);
    double x4 = cell->vertex(3)(0);

    double y1 = cell->vertex(0)(1);
    double y2 = cell->vertex(1)(1);
    double y3 = cell->vertex(2)(1);
    double y4 = cell->vertex(3)(1);

    double f1 = solution(cell->vertex_dof_index (0, 0));
    double f2 = solution(cell->vertex_dof_index (1, 0));
    double f3 = solution(cell->vertex_dof_index (2, 0));
    double f4 = solution(cell->vertex_dof_index (3, 0));

    double b=-(x1*y1*y2*f3-x1*y1*y2*f4-x1*y1*f3*y4+x1*y1*y4*f2-x1*y1*f2*y3+x1*y1*y3*f4+y3*x3*y2*f4-y2*x3*y3*f1+y3*x4*y4*f2-y3*x4*y4*f1+x3*y3*f1*y4-x3*y3*f2*y4+f3*x2*y2*y4-x2*y2*f1*y4+x2*y2*f1*y3-y3*x2*y2*f4-y1*x4*y4*f2-y1*y3*x3*f4+y1*x2*y2*f4+y1*x3*y3*f2-y1*x2*y2*f3-f3*y2*x4*y4+y1*f3*y4*x4+f1*y2*x4*y4)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double c=(x1*x2*y2*f4-x1*f3*x2*y2+x3*x1*y1*f4-x1*y1*x2*f4+x1*y1*x2*f3-x1*x4*y4*f2+x4*x1*y1*f2+x1*f3*y4*x4-x4*x1*y1*f3+x1*x3*y3*f2-x3*x1*y1*f2-x1*y3*x3*f4-x3*x2*y2*f4+x3*x2*y2*f1-x4*x2*y2*f1+x4*x2*y2*f3-f3*y4*x4*x2+x2*x4*y4*f1+y3*x3*x2*f4-x4*x3*y3*f2-x2*x3*y3*f1+x3*x4*y4*f2-x3*x4*y4*f1+x4*x3*y3*f1)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);
    double d=(-x3*y1*f4+x3*y1*f2+y1*f3*x4-x4*y1*f2-x1*y2*f4+x3*y2*f4-x3*f1*y2+y2*x1*f3+f1*y2*x4-f3*y2*x4+x4*y3*f2-x1*y3*f2-x4*y3*f1+x1*y4*f2-x3*y4*f2+x3*y4*f1-x2*f1*y4+f3*x2*y4+x2*f1*y3-y3*x2*f4+x1*y3*f4-x1*f3*y4+y1*x2*f4-y1*x2*f3)/(-y1*y3*x3*x2-x4*x2*y2*y1-x3*y3*x4*y2+x2*y3*x3*y4-x2*y2*x3*y4+x2*y1*x4*y4-x2*y3*x4*y4+y3*x4*x2*y2+y1*x3*x2*y2+y2*x3*x4*y4-y1*x3*x4*y4+x1*y1*x3*y4+y2*x1*y3*x3-x1*y4*x2*y1+x1*x2*y2*y4-y2*x3*x1*y1+y3*y1*x3*x4+x1*y3*x4*y4-x1*y3*x2*y2+x1*y1*y3*x2-x1*x3*y3*y4-x1*y3*x4*y1-x1*y2*x4*y4+y2*x4*x1*y1);

    grad(0) = b + d * cell->vertex(index_vertex)(1);
    grad(1) = c + d * cell->vertex(index_vertex)(0);

    return grad;
};

int main(int argc, char *argv[])
{
    const uint8_t dim = 2;

    const uint8_t xx = 0;
    const uint8_t yy = 1;
    const uint8_t xy = 2;

    typename HeatConductionProblemSup<dim>::TypeCoef coef;
    Femenist::Function<double, dim> rhsv;
//    Femenist::Function<double, dim> bound;

    coef[xx] .push_back (1.0);//(1.0/(2*1.25));//(E / (2 * (1 - Nu)));
    coef[yy] .push_back (1.0);//(1.0/(2*1.25));
    coef[xy] .push_back (0.0);
    coef[xx] .push_back (1.0);//(1.0/(2*1.25));
    coef[yy] .push_back (1.0);//(1.0/(2*1.25));
    coef[xy] .push_back (0.0);
    rhsv    = source<dim>;
//    bound   = boundary<dim>;

    Femenist::MyFuncFromDealii<dim>::Func NeumannBoundaryValues =
        [] (const dealii::Point<dim> &p) {return 0.0;};

    Femenist::MyFuncFromDealii<dim>::Func DBoundaryValues =
        [] (const dealii::Point<dim> &p) {
//    if ((p(0) > 2.0) and (p(0) < 6.0))
//        return p(0) / 2.0;
//    else
//        return p(0) / 1.0;
            return p(0)*p(0);//*p(0);
        };
//            return ((2 + Nu)*p(0)*p(0)*p(0)/6.0 - 
//                    (1.0/4.0 + (5.0/24.0)*Nu)*p(0) + 
//                    Nu/24.0);};

//    std::vector<Femenist::BoundaryValues<dim> > bound(1);
//    bound[0].function = NeumannBoundaryValues;
//    bound[0].boundari_indicator = 0;
//    bound[0].type = 1;
    std::vector<Femenist::BoundaryValues<dim> > bound(1);
    bound[0].function = DBoundaryValues;
    bound[0].boundari_indicator = 0;
    bound[0].type = 0;

    dealii::Triangulation<dim> tria;
//    ::set_circ(tria, 20.0, 4);
//    ::set_hexagon_grid_pure (tria, 100.0, 56.0);
//    ::set_single(tria);
//    ::set_grid_alternate(tria);
    ::set_grid(tria);
{
    std::ofstream out ("grid-igor.eps");
    dealii::GridOut grid_out;
    grid_out.write_eps (tria , out);
};
//tria .refine_global (1);
        typename dealii::Triangulation<dim>::active_face_iterator 
            face = tria.begin_active_face();
        typename dealii::Triangulation<dim>::active_face_iterator 
            end_face  = tria.end_face();
    
        for (; face != end_face; ++face)
{
    if (face->at_boundary())
    {
//        if (
//                ((fabs(face->vertex(0)(1) - 0.0) < 1e-10) and
//                 (fabs(face->vertex(1)(1) - 0.0) < 1e-10)) or
//                ((fabs(face->vertex(0)(1) - 8.0) < 1e-10) and
//                 (fabs(face->vertex(1)(1) - 8.0) < 1e-10)))
            face ->set_boundary_indicator (0);
    };
};

    class ::HeatConductionProblem<dim> hc_problem (tria, coef, bound, rhsv);

    REPORT hc_problem .solved ();

    hc_problem .print_result (std::string("res_"));

//    const size_t material_id_for_quadrate[4][4] =
//    {
//        {1, 1, 0, 0},
//        {1, 1, 0, 0},
//        {0, 0, 0, 0},
//        {0, 0, 0, 0}
//    };
//
//FILE *F;
//F = fopen ("mata-quadrate.gpd", "w");
//
//{
//    double s = 4;
//    while (s < 128)
//    {
//        dealii::Triangulation<2> tria;
//        const double dot[5] = 
//        {
//            (0.0),
//            (s / 2.0),
//            (s),
//            (s + (128.0 - s) / 2.0),
//            (128.0)
//        };
//
//        ::set_tria <5> (tria, dot, material_id_for_quadrate);
//
//        //    tria.begin()->face(0) ->set_boundary_indicator (1);
//        //    tria.begin()->face(1) ->set_boundary_indicator (1);
//
//        {
//            typename dealii::Triangulation<dim>::active_cell_iterator cell =
//                tria.begin_active();
//
//            typename dealii::Triangulation<dim>::active_cell_iterator endc =
//                tria.end();
//
//            for (; cell!=endc; ++cell)
//            {
//                FOR_I(0, 4)
//                {
//                    if (cell->face(i)->at_boundary())
//                        if (((cell->face(i)->vertex(0)[0] == 0.0) and
//                                    (cell->face(i)->vertex(1)[0] == 0.0)) or   
//                                ((cell->face(i)->vertex(0)[0] == 128.0) and
//                                 (cell->face(i)->vertex(1)[0] == 128.0)))   
//                            cell->face(i)->set_boundary_indicator(1);
//                };
//            };
//        };
//
//        tria .refine_global (3);
//
////    std::vector< dealii::Point< 2 > > v (4);
////    v[0][0] = -0.5; v[0][1] = -0.05;
////    v[1][0] = 0.5;  v[1][1] = -0.05;
////    v[2][0] = -0.5; v[2][1] = 0.05;
////    v[3][0] = 0.5;  v[3][1] = 0.05;
////
////    std::vector< dealii::CellData< 2 > > c (1, dealii::CellData<2>());
////    c[0].vertices[0] = 0;
////    c[0].vertices[1] = 1;
////    c[0].vertices[2] = 2;
////    c[0].vertices[3] = 3;
////    c[0].material_id = 0;
//
////    std::vector< dealii::Point< 2 > > v (6);
////
////    v[0][0] = 0.0; v[0][1] = 0.0;
////    v[1][0] = 1.0; v[1][1] = 0.0;
////    v[2][0] = 2.0; v[2][1] = 0.0;
////    v[3][0] = 0.0; v[3][1] = 2.0;
////    v[4][0] = 1.0; v[4][1] = 2.0;
////    v[5][0] = 2.0; v[5][1] = 2.0;
////
////    std::vector< dealii::CellData< 2 > > c (2, dealii::CellData<2>());
////
////    c[0].vertices[0] = 0;
////    c[0].vertices[1] = 1;
////    c[0].vertices[2] = 3;
////    c[0].vertices[3] = 4;
////    c[0].material_id = 0;
////
////    c[1].vertices[0] = 1;
////    c[1].vertices[1] = 2;
////    c[1].vertices[2] = 4;
////    c[1].vertices[3] = 5;
////    c[1].material_id = 1;
//
////    std::vector< dealii::Point< 1 > > v (3);
////
////    v[0][0] = 0.0; 
////    v[1][0] = 1.0;
////    v[2][0] = 2.0;
////
////    std::vector< dealii::CellData< 1 > > c (2, dealii::CellData<1>());
////
////    c[0].vertices[0] = 0;
////    c[0].vertices[1] = 1;
////    c[0].material_id = 0;
////
////    c[1].vertices[0] = 1;
////    c[1].vertices[1] = 2;
////    c[1].material_id = 1;
////
////    dealii::Triangulation<dim> tria;
////
////    tria .create_triangulation (v, c, dealii::SubCellData());
////
////    tria .refine_global (5);
//
//    class ::HeatConductionProblem<dim> hc_problem (tria, coef, bound, rhsv);
//
//    REPORT hc_problem .solved ();
//
//
//    std::vector<std::array<dealii::Point<dim>, 2> > 
//        grad_field(hc_problem.system_equations.x.size());
//
//    
//        typename dealii::DoFHandler<dim>::active_cell_iterator 
//            cell = hc_problem.domain.dof_handler.begin_active();
//        typename dealii::DoFHandler<dim>::active_cell_iterator 
//            end_cell  = hc_problem.domain.dof_handler.end();
//    
//        std::vector<uint8_t> divider(hc_problem.system_equations.x.size());
//        cell = hc_problem.domain.dof_handler.begin_active();
//        for (; cell != end_cell; ++cell)
//        {
//            for (size_t i = 0; i < 4; ++i)
//            {
//                grad_field[cell->vertex_dof_index(i,0)][0] = 
//                    cell->vertex(i);
//                grad_field[cell->vertex_dof_index(i,0)][1] +=
//                    ::get_grad<dim> (cell, hc_problem.system_equations.x, i); 
//                divider[cell->vertex_dof_index(i,0)] += 1;
//            };
//        };
//        for (size_t i = 0; i < divider.size(); ++i)
//            grad_field[i][1] /= divider[i];
//    };
//
//    double meta_coef = 0.0;
//    for (auto grad : grad_field)
//    {
////        if (
////                (grad[0](0) > dot[1] - 1e-5) and
////                (grad[0](1) > dot[1] - 1e-5) and
////                (grad[0](0) < dot[3] + 1e-5) and
////                (grad[0](1) < dot[3] + 1e-5)
////           )
////            meta_coef += grad[1](0);// * coef[xx][1];
////        else
////            meta_coef += grad[1](0);// * coef[xx][0];
//        if (fabs(grad[0](0) - 128.0) < 1e-10)
//            meta_coef += grad[1](0) * coef[xx][0];
//    };
////    FOR_I(0, grad_field.size())
////    {
////        meta
////    }
////    FOR_I(0, hc_problem.system_equations.x.size()) { 
////        meta_coef += hc_problem.system_equations.x(i) *
////                     (-hc_problem.system_equations.b(i));
////    };
//     
//    meta_coef /= 
////        grad_field.size();
//        33.0;
//
//    fprintf(F, "%f %f %f\n", s*s, meta_coef, s);
//    printf("%f %f\n", s*s, meta_coef);
//
//    s += 4.0;
//
////    hc_problem .print_result ("output.gpd");
//
////    for (size_t i = 0; i < grad_field.size(); ++i)
////        if (
////                (grad_field[i][0](0) > dot[1] - 1e-5) and
////                (grad_field[i][0](1) > dot[1] - 1e-5) and
////                (grad_field[i][0](0) < dot[3] + 1e-5) and
////                (grad_field[i][0](1) < dot[3] + 1e-5)
////           )
////           hc_problem .system_equations .x (i) = grad_field[i][1](1);// * coef[xx][1];
//////    puts("in");
////        else
//////           puts("out");
////           hc_problem .system_equations .x (i) = grad_field[i][1](0);// * coef[xx][0];
//////        hc_problem .system_equations .x (i) = grad_field[i][1][0];
////
////    hc_problem .print_result ("grad.gpd");
//    };
//};
//
//fclose(F);

//    printf("%f\n", meta_coef);

//    hc_problem .print_result ("output.gpd");

    return 0;
}
//
