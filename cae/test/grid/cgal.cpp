#include "stdio.h"
//#include "grid.cpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Polygon_2.h>
// #include <CGAL/algorithm.h>
// #include <CGAL/Constrained_Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Mesh_cell_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
// #include <CGAL/Delaunay_mesher_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_reordering.h>

#include "projects/cae/main/point/point.h"
// #include "projects/cae/test/file/file.h"

struct Border_id
{
    Border_id() : border_id(-1) {};
    int border_id;
};
struct Material_id
{
    Material_id() : material_id(-1) {};
    int material_id;
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;


typedef CGAL::Triangulation_vertex_base_with_info_2<Border_id,K> Vb;
//typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;

typedef CGAL::Triangulation_face_base_with_info_2<Material_id,K> Fbbb;
typedef CGAL::Constrained_triangulation_face_base_2<K,Fbbb>      Fbb;
typedef CGAL::Delaunay_mesh_face_base_2<K,Fbb>                   Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>             Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds>       CDT;

typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;


typedef CGAL::Triangulation_vertex_base_with_info_3<Border_id,K> Vb3;

typedef CGAL::Triangulation_cell_base_with_info_3<Material_id,K> Fbbb3;
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;
typedef CGAL::Mesh_cell_base_3<K, Mesh_domain, Fbbb3>                   Fb3;
typedef CGAL::Triangulation_data_structure_3<Vb3, Fbbb3>             Tds3;
typedef CGAL::Delaunay_triangulation_3<K, Tds3>       CDT3;
typedef CGAL::Mesh_complex_3_in_triangulation_3<CDT3> C3t3;


typedef CDT::Vertex_handle Vertex_handle;
// typedef CDT::Point Point;
void create_grid (CDT &cdt,
        const vec<prmt::Point<2>> &outer_border,
        const vec<prmt::Point<2>> &inner_border)
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
    // 
    for(auto it = vec_of_domains.begin(); it != vec_of_domains.end(); ++it) 
    {
        // auto it = vec_of_domains.begin();
        // ++it;
       for(auto vit = it->begin() + 1; vit != it->end(); ++vit)
       {
          cdt.insert_constraint( *(vit - 1), *vit );
       }
       cdt.insert_constraint( *( it->end() - 1), *( it->begin() ) );
    }
    
    for(auto it = cdt.finite_faces_begin(); it != cdt.finite_faces_end(); ++it) 
    {
    it->info().material_id = 10;
    };
    

    std::list<CDT::Point> list_of_seeds;
    list_of_seeds.push_back(CDT::Point(0.5, 0.5));
    // // std::list<Point> list_of_seeds_1;
    // // list_of_seeds_1.push_back(Point(0.5, 0.5));
    // // list_of_seeds_1.push_back(Point(0.2, 0.2));
    // // std::list<Point> list_of_seeds_2;
    // // list_of_seeds_2.push_back(Point(0.5, 0.5));

    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
            Criteria(), true);
    // CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
    //                           Criteria());

    for(auto it = cdt.finite_faces_begin(); it != cdt.finite_faces_end(); ++it) 
    {
        // printf("%d\n", it->info().id);
    }

    {
        FILE* F;
        F = fopen ("out.gpd", "w");
        // for(auto it = cdt.finite_vertices_begin(); it != cdt.finite_vertices_end(); ++it) 
        // {
        //  fprintf(F, "%f %f\n", it->point().x(), it->point().y());
        // };        
        // fclose(F);
        // };
        for(auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
        {
            CDT::Triangle trg = cdt.triangle(fit);
            fprintf(F, "%f %f\n", trg[0].x(), trg[0].y());
            fprintf(F, "%f %f\n", trg[1].x(), trg[1].y());
            fprintf(F, "%f %f\n", trg[2].x(), trg[2].y());
        };
        fclose(F);
    };

    CDT::Locate_type loc;
    int li = 0;


        puts(" ");
    CDT::Triangle trg = cdt.triangle(cdt.locate(CDT::Point(0.0, 0.0), loc, li));
        printf("t %f %f\n", trg[0].x(), trg[0].y());
        printf("t %f %f\n", trg[1].x(), trg[1].y());
        printf("t %f %f\n", trg[2].x(), trg[2].y());

        if (loc == CDT::VERTEX) puts("vertex");
        if (loc == CDT::EDGE) puts("edge");
        if (loc == CDT::FACE) puts("face");
};

void d2_color ()
{
    CDT cdt, cdt_no_refine;

    cdt_no_refine.insert(CDT::Point(0.0, 0.0));
    auto c1 = cdt_no_refine.insert(CDT::Point(1.0, 0.0));
    auto c2 = cdt_no_refine.insert(CDT::Point(0.0, 1.0));
    cdt_no_refine.insert(CDT::Point(1.0, 1.0));

    cdt.insert_constraint(c1, c2);

    cdt_no_refine.locate(CDT::Point(0.25, 0.25))->info().material_id = 10;
    cdt_no_refine.locate(CDT::Point(0.75, 0.75))->info().material_id = 20;

    CDT::Locate_type loc;
    int li = 0;
    {
        auto h = cdt_no_refine.locate(CDT::Point(0.0, 0.0), loc, li);
        h->vertex(li)->info().border_id = 1;
            printf("%d\n", loc);
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(1.0, 0.0), loc, li);
        h->vertex(li)->info().border_id = 2;
            printf("%d\n", loc);
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(1.0, 1.0), loc, li);
        h->vertex(li)->info().border_id = 3;
            printf("%d\n", loc);
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(0.0, 1.0), loc, li);
        h->vertex(li)->info().border_id = 4;
            printf("%d\n", loc);
    };

    cdt = cdt_no_refine;
    // cdt.insert(CDT::Point(0.25, 0.25));
    // cdt.insert(CDT::Point(0.75, 0.75));
    // cdt.insert(CDT::Point(0.6, 0.6));
    // cdt.insert(CDT::Point(0.4, 0.1));
    // cdt.insert(CDT::Point(0.4, 0.0));
    // cdt.insert(CDT::Point(1.0, 0.5));
    // cdt.insert_constraint(cdt.insert(CDT::Point(0.6, 0.6)), cdt.insert(CDT::Point(0.85, 0.85)));
    {
    auto p1 = cdt.insert(CDT::Point(0.6, 0.6));
    auto p2 = cdt.insert(CDT::Point(0.8, 0.6));
    auto p3 = cdt.insert(CDT::Point(0.8, 0.8));
    auto p4 = cdt.insert(CDT::Point(0.6, 0.8));
    cdt.insert_constraint(p1, p2);
    cdt.insert_constraint(p2, p3);
    cdt.insert_constraint(p3, p4);
    cdt.insert_constraint(p4, p1);
    };

    CGAL::refine_Delaunay_mesh_2(cdt, Criteria( 0.125, 0.1 ));

    {
    };

    // CGAL::refine_Delaunay_mesh_2(cdt, Criteria( 0.125, 0.1 ));

    {
    };

    // CGAL::refine_Delaunay_mesh_2(cdt, Criteria( 0.125, 0.1 ));

    {
    };

    // CGAL::refine_Delaunay_mesh_2(cdt, Criteria( 0.125, 0.1 ));

    {
        auto f = fopen("area.out", "w");
    // CGAL::refine_Delaunay_mesh_2(cdt, Criteria(0.125, 0));
    for(auto it = cdt.finite_faces_begin(); it != cdt.finite_faces_end(); ++it) 
    {
        CDT::Point p;
        {   //sum(0,2)([trg](){return trg[i].x()});
            CDT::Triangle trg = cdt.triangle(it);
            auto x = (trg[0].x() + trg[1].x() + trg[2].x()) / 3.0;
            auto y = (trg[0].y() + trg[1].y() + trg[2].y()) / 3.0;
            p = CDT::Point(x,y);
        };
        printf("%f %f\n", p.x(), p.y());
        auto h = cdt_no_refine.locate(p);
        printf("%d\n", h->info().material_id);
        CDT::Triangle trg = cdt.triangle(it);
        printf("t (%f %f) (%f %f) (%f %f)\n", 
                trg[0].x(), trg[0].y(), trg[1].x(), trg[1].y(), trg[2].x(), trg[2].y());
        fprintf(f, "%f %f %f|%f %f %f|%s\n", 
                trg[0].x(), trg[1].x(), trg[2].x(),
                trg[0].y(), trg[1].y(), trg[2].y(),
                h->info().material_id == 10 ? "#ff0000" : "#0000ff");
    };
    fclose(f);
    puts(" ");
    };
    
    {
        auto f = fopen("border.out", "w");
    auto bv = cdt.incident_vertices (cdt.infinite_vertex());
    auto bv_end = bv;
    do
    {
        auto p1 = bv->point();
        printf("%f %f\n", p1.x(), p1.y());
        // ++bv;
        auto p2 = (++bv)->point();
        // // --bv;
        auto p = CDT::Point((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0);
        // printf("%f %f\n", p.x(), p.y());
        auto h = cdt_no_refine.locate(p, loc, li);
        fprintf(f, "%f %f|%f %f|#00%s00\n", 
                p1.x(), p2.x(),
                p1.y(), p2.y(),
                h->vertex(h->ccw(li))->info().border_id != 1 ?
                h->vertex(h->ccw(li))->info().border_id != 2 ?
                h->vertex(h->ccw(li))->info().border_id != 3 ?
                "ff" : "aa" : "55" : "00"
                );
        printf("%d\n", h->vertex(h->ccw(li))->info().border_id);
        printf("%d %d %f %f\n", loc, 
                h->vertex(h->ccw(li))->info().border_id,
                h->vertex(h->ccw(li))->point().x(),
                h->vertex(h->ccw(li))->point().y());
    } while (bv != bv_end);
    fclose(f);
    puts(" ");
    };

    // for(auto it = cdt.finite_vertices_begin(); it != cdt.finite_vertices_end(); ++it) 
    // {
    //     printf("%d\n", it->info().border_id);
    // };
    {
        auto h = cdt_no_refine.locate(CDT::Point(0.5, 0.0), loc, li);
        printf("%d %d %f %f\n", loc, 
                h->vertex(h->ccw(li))->info().border_id,
                h->vertex(h->ccw(li))->point().x(),
                h->vertex(h->ccw(li))->point().y());
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(1.0, 0.5), loc, li);
        printf("%d %d %f %f\n", loc, 
                h->vertex(h->ccw(li))->info().border_id,
                h->vertex(h->ccw(li))->point().x(),
                h->vertex(h->ccw(li))->point().y());
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(0.5, 1.0), loc, li);
        printf("%d %d %f %f\n", loc, 
                h->vertex(h->ccw(li))->info().border_id,
                h->vertex(h->ccw(li))->point().x(),
                h->vertex(h->ccw(li))->point().y());
    };
    {
        auto h = cdt_no_refine.locate(CDT::Point(0.0, 0.5), loc, li);
        printf("%d %d %f %f\n", loc, 
                h->vertex(h->ccw(li))->info().border_id,
                h->vertex(h->ccw(li))->point().x(),
                h->vertex(h->ccw(li))->point().y());
    };


};

void multi_inclusion_test ()
{
    CDT cdt;

    cdt.insert(CDT::Point(0.0, 0.0));
    auto c1 = cdt.insert(CDT::Point(10.0, 0.0));
    auto c2 = cdt.insert(CDT::Point(0.0, 10.0));
    cdt.insert(CDT::Point(10.0, 10.0));

    // cdt.insert_constraint(c1, c2);

    std::list<CDT::Point> list_of_seeds;
    list_of_seeds.push_back(CDT::Point(0.25, 0.25));

    CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
                               Criteria( 0.125, 0.1 ), true);
    // CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
    //                            Criteria(0.125, 0.1));
    // CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
    //                            Criteria(), true);
    // CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
    //                            Criteria());

    {
        auto f = fopen("area.out", "w");
    // CGAL::refine_Delaunay_mesh_2(cdt, Criteria(0.125, 0));
    for(auto it = cdt.finite_faces_begin(); it != cdt.finite_faces_end(); ++it) 
    {
        CDT::Triangle trg = cdt.triangle(it);
        printf("t (%f %f) (%f %f) (%f %f)\n", 
                trg[0].x(), trg[0].y(), trg[1].x(), trg[1].y(), trg[2].x(), trg[2].y());
        fprintf(f, "%f %f %f|%f %f %f|%s\n", 
                trg[0].x(), trg[1].x(), trg[2].x(),
                trg[0].y(), trg[1].y(), trg[2].y(),
                0 ? "#ff0000" : "#0000ff");
    };
    fclose(f);
    };
};

int main ()
{        
    // C3t3::Triangulation t;
    // t.insert(CDT3::Point(0,0,0));
    // t.insert(CDT3::Point(1,0,0));
    // t.insert(CDT3::Point(0,1,0));
    // t.insert(CDT3::Point(0,0,1));
    // t.insert(CDT3::Point(2,2,2));
    // t.insert(CDT3::Point(-1,0,1));
    // C3t3 c3t3 = t;
    // c3t3.insert(CDT3::Point(0,0,0));
    // Mesh_domain domain(t);
    // CGAL::Implicit_mesh_domain_3<Function,K> criteria(facet_angle=30, facet_size=0.5, facet_distance=0.025,
    //                                  cell_radius_edge_ratio=2, cell_size=0.1);
    vec<prmt::Point<2>> outer_border;
    vec<prmt::Point<2>> inner_border;
    
    outer_border .push_back (prmt::Point<2>(1.0,1.0));
    outer_border .push_back (prmt::Point<2>(1.0,3.0));
    outer_border .push_back (prmt::Point<2>(3.0,3.0));
    outer_border .push_back (prmt::Point<2>(3.0,1.0));
    
    inner_border .push_back (prmt::Point<2>(0.0,0.0));
    inner_border .push_back (prmt::Point<2>(0.0,4.0));
    inner_border .push_back (prmt::Point<2>(4.0,4.0));
    inner_border .push_back (prmt::Point<2>(4.0,0.0));

    CDT cdt;
    // create_grid(cdt, outer_border, inner_border);
    //create_grid_using_cgal<cguc_type::Default>(
            //cdt, outer_border, inner_border);
            
    // std::ofstream outf("tria.cgal", std::ios::out);
    // cdt.tds().file_output (outf);
    // outf.close();
    // puts("sdfsdf");

    // d2_color();
    // printf("%d\n", CDT::EDGE);
    multi_inclusion_test ();

return 0;
}
