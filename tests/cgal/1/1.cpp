#include <iostream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
int main()
{
  Point_2 points[6] = { Point_2(0,0), Point_2(1,1), Point_2(10,10), Point_2(10,0), Point_2(1,2), Point_2(2,2) };
  Point_2 result[6];
  Point_2 *ptr = CGAL::convex_hull_2( points, points+5, result );
  std::cout <<  ptr - result << " points on the convex hull" << std::endl;
  for (auto p : points)
  	std::cout << p.x() << ' ' << p.y() << std::endl;
  	puts(" ");
  for (auto p : result)
  	std::cout << p.x() << ' ' << p.y() << std::endl;
  return 0;
}
