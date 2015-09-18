
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

#include "../../../calculation_core/src/blocks/general/domain/domain.h"
#include "../../../calculation_core/src/blocks/general/laplacian/scalar/laplacian_scalar.h"
#include "../../../calculation_core/src/blocks/general/source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/general/boundary_value/boundary_value.h"
#include "../../../calculation_core/src/blocks/general/assembler/assembler.h"
#include "../../../calculation_core/src/blocks/general/system_linear_algebraic_equations/system_linear_algebraic_equations.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/trivial_prepare_system_equations/trivial_prepare_system_equations.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/apply_boundary_value/scalar/apply_boundary_value_scalar.h"
#include "../../../calculation_core/src/blocks/general/geometric_tools/geometric_tools.h"
#include "../../../calculation_core/src/blocks/special/heat_conduction_problem_tools/heat_conduction_problem_tools.h"

#include "../../../calculation_core/src/blocks/special/problem_on_cell/source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations/prepare_system_equations.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/prepare_system_equations_with_cubic_grid/prepare_system_equations_with_cubic_grid.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/system_linear_algebraic_equations/system_linear_algebraic_equations.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/calculate_meta_coefficients/calculate_meta_coefficients.h"
#include "../../../calculation_core/src/blocks/special/problem_on_cell/assembler/assembler.h"

#include "../../../calculation_core/src/blocks/general/laplacian/vector/laplacian_vector.h"
#include "../../../calculation_core/src/blocks/general/source/vector/source_vector.h"
#include "../../../calculation_core/src/blocks/general/additional_tools/apply_boundary_value/vector/apply_boundary_value_vector.h"
#include "../../../calculation_core/src/blocks/special/elastic_problem_tools/elastic_problem_tools.h"

#include "../../../calculation_core/src/blocks/special/problem_on_cell/source/vector/source_vector.h"


#include "../../../calculation_core/src/blocks/special/feature_source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/feature_source/vector/source_vector.h"

#include "../../../calculation_core/src/blocks/special/poly_materials_source/scalar/source_scalar.h"
#include "../../../calculation_core/src/blocks/special/poly_materials_source/vector/source_vector.h"

#include "../../../calculation_core/src/blocks/general/4_points_function/4_points_function.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_in.h>
