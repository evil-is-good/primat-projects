#ifndef ELASTIC_CELL_CALCULATOR
#define ELASTIC_CELL_CALCULATOR 1

#include "pch.h"

str make_path(const vec<str>& path)
{
    str partial_path = "/home/primat/hoz_block_disk/cell_func_bd/";
    for (auto&& folder : path)
    {
        partial_path += folder;
        if (access(partial_path.c_str(), 0))
        {
            mkdir(partial_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        };
        partial_path += "/";
    };

    return partial_path;
};

void set_cylinder(dealii::Triangulation< 3 > &triangulation, 
        const double radius, cst ort, const size_t n_refine)
{
    dealii::GridGenerator ::hyper_cube (triangulation, 0.0, 1.0);
    triangulation .refine_global (n_refine);
    {
        dealii::Point<2> center (0.5, 0.5);
        dealii::Triangulation<3>::active_cell_iterator
            cell = triangulation .begin_active(),
                 end_cell = triangulation .end();
        for (; cell != end_cell; ++cell)
        {
            dealii::Point<2> midle_p(0.0, 0.0);

            for (size_t i = 0; i < 8; ++i)
            {
                st count = 0;
                for (st j = 0; j < 3; ++j)
                {
                    if (j != ort)
                    {
                        midle_p(count) += cell->vertex(i)(j);
                        ++count;
                    };
                };
            };
            midle_p(0) /= 8.0;
            midle_p(1) /= 8.0;


            if (center.distance(midle_p) < radius)
            {
                cell->set_material_id(1);
            }
            else
                cell->set_material_id(0);
        };
    };
};

void solve_cell_elastic_problem (
        const str& path, cst number_of_approx, 
        const vec<ATools::FourthOrderTensor>& C, Domain<3>& domain)
{
    enum {x, y, z};

    dealii::FESystem<3,3> fe (dealii::FE_Q<3,3>(1), 3);
    domain.dof_init (fe);

    OnCell::SystemsLinearAlgebraicEquations<1> slae;
    OnCell::BlackOnWhiteSubstituter bows;

    LaplacianVector<3> element_matrix (domain.dof_handler.get_fe());
    element_matrix.C .resize (2);
    for (st i = 0; i < 3; ++i)
    {
        for (st j = 0; j < 3; ++j)
        {
            for (st k = 0; k < 3; ++k)
            {
                for (st l = 0; l < 3; ++l)
                {
                    for (st m = 0; m < C.size(); ++m)
                    {
                        element_matrix.C[m][i][j][k][l] = C[m][i][j][k][l];
                    };
                };
            };
        };
    };
        // EPTools ::set_isotropic_elascity{yung : 1.0, puasson : 0.2}(element_matrix.C[0]);
        // EPTools ::set_isotropic_elascity{yung : 10.0, puasson : 0.28}(element_matrix.C[1]);

    OnCell::prepare_system_equations_with_cubic_grid<3, 3> (slae, bows, domain);

    OnCell::Assembler::assemble_matrix<3> (slae.matrix, element_matrix, domain.dof_handler, bows);

    // cst number_of_approx = 3; // Начиная с нулевой
    OnCell::ArrayWithAccessToVector<arr<arr<dbl, 3>, 3>> meta_coefficient(number_of_approx+1);
    OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> cell_func (number_of_approx);
    OnCell::ArrayWithAccessToVector<arr<dealii::Vector<dbl>, 3>> N_func (number_of_approx);
    OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_stress (number_of_approx);
    OnCell::ArrayWithAccessToVector<arr<arr<dealii::Vector<dbl>, 3>, 3>> cell_deform (number_of_approx);
    OnCell::ArrayWithAccessToVector<arr<arr<arr<dbl, 3>, 3>, 3>> true_meta_coef (number_of_approx);
    for (auto &&a : meta_coefficient.content)
        for (auto &&b : a)
            for (auto &&c : b)
                for (auto &&d : c)
                    for (auto &&e : d)
                        e = 0.0;
    for (auto &&a : cell_func.content)
        for (auto &&b : a)
            for (auto &&c : b)
                for (auto &&d : c)
                    d .reinit (slae.solution[0].size());
    // Нулевое приближение ячейковой функции для которой nu==aplha равно 1.0
    for (st i = 0; i < slae.solution[0].size(); ++i)
    {
        if ((i % 3) == x) cell_func[arr<i32, 3>{0, 0, 0}][x][i] = 1.0;
        if ((i % 3) == y) cell_func[arr<i32, 3>{0, 0, 0}][y][i] = 1.0;
        if ((i % 3) == z) cell_func[arr<i32, 3>{0, 0, 0}][z][i] = 1.0;
    };
    for (auto &&a : N_func.content)
        for (auto &&b : a)
            for (auto &&c : b)
                for (auto &&d : c)
                    d .reinit (slae.solution[0].size());
    for (auto &&a : cell_stress.content)
        for (auto &&b : a)
            for (auto &&c : b)
                for (auto &&d : c)
                    for (auto &&e : d)
                        e .reinit (slae.solution[0].size());
    for (auto &&a : cell_deform.content)
        for (auto &&b : a)
            for (auto &&c : b)
                for (auto &&d : c)
                    for (auto &&e : d)
                        e .reinit (slae.solution[0].size());


    OnCell::MetaCoefficientElasticCalculator mc_calculator (
            domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());

    for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    {
        for (i32 i = 0; i < approx_number+1; ++i)
        {
            for (i32 j = 0; j < approx_number+1; ++j)
            {
                for (i32 k = 0; k < approx_number+1; ++k)
                {
                    if ((i+j+k) == approx_number)
                    {
                        arr<i32, 3> approximation = {i, j, k};
                        for (st nu = 0; nu < 3; ++nu)
                        {
                            slae.solution[0] = 0.0;
                            slae.rhsv[0] = 0.0;

                            OnCell::SourceVectorApprox<3> element_rhsv (approximation, nu,
                                    element_matrix.C, 
                                    meta_coefficient,
                                    cell_func,
                                    // &psi_func,
                                    domain.dof_handler.get_fe());
                            OnCell::Assembler::assemble_rhsv<3> (slae.rhsv[0], element_rhsv, domain.dof_handler, bows);

                            dealii::SolverControl solver_control (500000, 1e-12);
                            dealii::SolverCG<> solver (solver_control);
                            solver.solve (
                                    slae.matrix,
                                    slae.solution[0],
                                    slae.rhsv[0]
                                    ,dealii::PreconditionIdentity()
                                    );
                            FOR(i, 0, slae.solution[0].size())
                                slae.solution[0][i] = slae.solution[0][bows.subst (i)];
                            FOR(i, 0, slae.rhsv[0].size())
                                slae.rhsv[0][i] = slae.rhsv[0][bows.subst (i)];

                            cell_func[approximation][nu] = slae.solution[0];
                            // N_func[approximation][nu] = slae.rhsv[0];
                        };
                    };
                };
            };
        };
        puts("!!!");
        for (i32 i = 0; i < approx_number+2; ++i)
        {
            for (i32 j = 0; j < approx_number+2; ++j)
            {
                for (i32 k = 0; k < approx_number+2; ++k)
                {
                    if ((i+j+k) == approx_number+1)
                    {
                        arr<i32, 3> approximation = {i, j, k};
                        for (st nu = 0; nu < 3; ++nu)
                        {
                            auto res = mc_calculator .calculate (
                                    approximation, nu,
                                    domain.dof_handler, cell_func);
                            meta_coefficient[approximation][nu][x] = res[x]; //E_x_a[0]_nu_a[1]
                            meta_coefficient[approximation][nu][y] = res[y]; 
                            meta_coefficient[approximation][nu][z] = res[z]; 
                        };
                    };
                };
            };
        };
    };
    puts("!!!!!");

    // {
    //     arr<str, 3> ort = {"x", "y", "z"};
    //     arr<str, 3> aprx = {"0", "1", "2"};
    //     OnCell::StressCalculator stress_calculator (
    //             domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe());
    //     for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    //     {
    //         for (st i = 0; i < approx_number+1; ++i)
    //         {
    //             for (st j = 0; j < approx_number+1; ++j)
    //             {
    //                 for (st k = 0; k < approx_number+1; ++k)
    //                 {
    //                     if ((i+j+k) == approx_number)
    //                     {
    //                         arr<i32, 3> approximation = {i, j, k};
    //                         for (st nu = 0; nu < 3; ++nu)
    //                         {
    //                             for (st alpha = 0; alpha < 3; ++alpha)
    //                             {
    //                                 dealii::Vector<dbl> stress(domain.dof_handler.n_dofs());
    //                                 for (st beta = 0; beta < 3; ++beta)
    //                                 {
    //                                     stress_calculator .calculate (
    //                                             approximation, nu, alpha, beta,
    //                                             domain.dof_handler, cell_func, stress);
    //                                     true_meta_coef[approximation][nu][alpha][beta] =
    //                                         OnCell::calculate_meta_coefficients_3d_elastic_from_stress (
    //                                                 domain.dof_handler, stress, beta);
    //                                     // printf("meta k=(%ld, %ld, %ld) nu=%ld alpha=%ld beta=%ld %f\n", 
    //                                     //         i, j, k, nu, alpha, beta, true_meta_coef[approximation][nu][alpha][beta]);
    //                                 };
    //                                 cell_stress[approximation][nu][alpha] = stress;
    //                             };
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    // {
    //     arr<str, 3> ort = {"x", "y", "z"};
    //     arr<str, 3> aprx = {"0", "1", "2"};
    //     OnCell::DeformCalculator deform_calculator (
    //             domain.dof_handler, domain.dof_handler.get_fe());
    //     for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    //     {
    //         for (st i = 0; i < approx_number+1; ++i)
    //         {
    //             for (st j = 0; j < approx_number+1; ++j)
    //             {
    //                 for (st k = 0; k < approx_number+1; ++k)
    //                 {
    //                     if ((i+j+k) == approx_number)
    //                     {
    //                         arr<i32, 3> approximation = {i, j, k};
    //                         for (st nu = 0; nu < 3; ++nu)
    //                         {
    //                             dealii::Vector<dbl> deform(domain.dof_handler.n_dofs());
    //                             for (st beta = 0; beta < 3; ++beta)
    //                             {
    //                                 deform_calculator .calculate (
    //                                         approximation, nu, beta,
    //                                         domain.dof_handler, cell_func, deform);
    //                                 cell_deform[approximation][nu][beta] = deform;
    //                             };
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    //
    // ATools::FourthOrderTensor meta_coef;
    // for (st i = 0; i < 3; ++i)
    // {
    //     for (st j = 0; j < 3; ++j)
    //     {
    //         for (st k = 0; k < 3; ++k)
    //         {
    //             meta_coef[j][k][i][x] = true_meta_coef[arr<i32, 3>{1, 0, 0}][i][j][k];
    //             meta_coef[j][k][i][y] = true_meta_coef[arr<i32, 3>{0, 1, 0}][i][j][k];
    //             meta_coef[j][k][i][z] = true_meta_coef[arr<i32, 3>{0, 0, 1}][i][j][k];
    //         };
    //     };
    // };

    // auto newcoef = unphysical_to_physicaly (meta_coef);
    // printf("%f %f %f %f %f %f %f %f %f %f %f\n", 
    //         newcoef[0][0][0][0],
    //         newcoef[0][0][1][1],
    //         newcoef[0][0][2][2],
    //         newcoef[1][1][0][0],
    //         newcoef[1][1][1][1],
    //         newcoef[1][1][2][2],
    //         newcoef[2][2][0][0],
    //         newcoef[2][2][1][1],
    //         newcoef[2][2][2][2],
    //         meta_coef[0][1][0][1],
    //         meta_coef[0][2][0][2]
    //       );


    // {
    //     std::ofstream out (path + "/meta_coef.bin", std::ios::out | std::ios::binary);
    //     out.write ((char *) &meta_coef, sizeof meta_coef);
    //     out.close ();
    // };
    // OnCell::BDStore::print_coor<3> (domain.dof_handler, path + "/coor");
    OnCell::BDStore::print_stress (
                domain.dof_handler, element_matrix.C, domain.dof_handler.get_fe(),
                cell_func, number_of_approx, path);

    // arr<str, 3> ort = {"x", "y", "z"};
    // arr<str, 3> aprx = {"0", "1", "2"};
    // for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    // {
    //     for (st i = 0; i < approx_number+1; ++i)
    //     {
    //         for (st j = 0; j < approx_number+1; ++j)
    //         {
    //             for (st k = 0; k < approx_number+1; ++k)
    //             {
    //                 if ((i+j+k) == approx_number)
    //                 {
    //                     arr<i32, 3> approximation = {i, j, k};
    //                     for (st nu = 0; nu < 3; ++nu)
    //                     {
    //                         for (st alpha = 0; alpha < 3; ++alpha)
    //                         {
    //                             str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
    //                             {
    //                                 std::ofstream out (path+"/stress_"+name+".bin", std::ios::out | std::ios::binary);
    //                                 for (st i = 0; i < slae.solution[0].size(); ++i)
    //                                 {
    //                                     out.write ((char *) &(cell_stress[approximation][nu][alpha][i]), sizeof(dbl));
    //                                 };
    //                                 out.close ();
    //                                 // EPTools ::print_move_slice (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
    //                                 //         "cell/"+f_name+"/stress_"+name+".gpd", y, 0.5);
    //                                 // EPTools ::print_move<3> (cell_stress[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
    //                                 //         "cell/"+f_name+"/stress_"+name+".vtk", dealii::DataOutBase::vtk);
    //                             };
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    //
    // for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    // {
    //     for (st i = 0; i < approx_number+1; ++i)
    //     {
    //         for (st j = 0; j < approx_number+1; ++j)
    //         {
    //             for (st k = 0; k < approx_number+1; ++k)
    //             {
    //                 if ((i+j+k) == approx_number)
    //                 {
    //                     arr<i32, 3> approximation = {i, j, k};
    //                     for (st nu = 0; nu < 3; ++nu)
    //                     {
    //                         for (st alpha = 0; alpha < 3; ++alpha)
    //                         {
    //                             str name = aprx[i]+str("_")+aprx[j]+str("_")+aprx[k]+str("_")+ort[nu]+str("_")+ort[alpha];
    //                             {
    //                                 std::ofstream out ("cell/"+f_name+"/deform_"+name+".bin", std::ios::out | std::ios::binary);
    //                                 for (st i = 0; i < slae.solution[0].size(); ++i)
    //                                 {
    //                                     out.write ((char *) &(cell_deform[approximation][nu][alpha][i]), sizeof(dbl));
    //                                 };
    //                                 out.close ();
    //                                 EPTools ::print_move_slice (cell_deform[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
    //                                         "cell/"+f_name+"/deform_"+name+".gpd", y, 0.5);
    //                                 EPTools ::print_move<3> (cell_deform[arr<i32,3>{i,j,k}][nu][alpha], domain.dof_handler, 
    //                                         "cell/"+f_name+"/deform_"+name+".vtk", dealii::DataOutBase::vtk);
    //                             };
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    //
    // for (st approx_number = 1; approx_number < number_of_approx; ++approx_number)
    // {
    //     for (st i = 0; i < approx_number+1; ++i)
    //     {
    //         for (st j = 0; j < approx_number+1; ++j)
    //         {
    //             for (st k = 0; k < approx_number+1; ++k)
    //             {
    //                 if ((i+j+k) == approx_number)
    //                 {
    //                     arr<i32, 3> approximation = {i, j, k};
    //                     for (st nu = 0; nu < 3; ++nu)
    //                     {
    //                         str name = ort[i]+str("_")+ort[j]+str("_")+ort[k]+str("_")+ort[nu];
    //                         {
    //                             std::ofstream out ("cell/"+f_name+"/solution_"+name+".bin", std::ios::out | std::ios::binary);
    //                             for (st i = 0; i < slae.solution[0].size(); ++i)
    //                             {
    //                                 out.write ((char *) &(cell_func[approximation][nu][i]), sizeof(dbl));
    //                             };
    //                             out.close ();
    //                         };
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    //
    // {
    //     std::ofstream out ("cell/"+f_name+"/solution_on_cell_size.bin", std::ios::out | std::ios::binary);
    //     auto size = slae.solution[0].size();
    //     out.write ((char *) &size, sizeof size);
    //     out.close ();
    // };
};

void solve_cell_elastic_problem_circle(
        const str& path, cst approximation_number, const vec<arr<dbl, 6>>& prop, cst refine, cdbl R)
{
    enum {x, y, z};

    Domain<3> domain;
    set_cylinder(domain.grid, R, y, refine);
    vec<ATools::FourthOrderTensor> C(prop.size());
    for (st i = 0; i < prop.size(); ++i)
    {
        EPTools ::set_isotropic_elascity{yung : prop[i][0], puasson : prop[i][3]}(C[i]);
    };
    // auto C2d = t4_to_t2 (element_matrix.C[0]);
    //     printf("C2d\n");
    //     for (size_t i = 0; i < 6; ++i)
    //     {
    //         for (size_t j = 0; j < 6; ++j)
    //         {
    //             if (std::abs(C2d[i][j]) > 0.0000001)
    //                 printf("\x1B[31m%f\x1B[0m   ", 
    //                         C2d[i][j]);
    //             else
    //                 printf("%f   ", 
    //                         C2d[i][j]);
    //         };
    //         for (size_t i = 0; i < 2; ++i)
    //             printf("\n");
    //     };
    //     printf("\n");
    // };

    // solve_approx_cell_elastic_problem (path, approximation_number, C, domain);
    solve_cell_elastic_problem (path, approximation_number, C, domain);
};

int main(int argc, char *argv[])
{
    if (argc == 17)
    {
        vec<arr<dbl, 6>> prop(atoi(argv[1]));

        for (st i = 0; i < prop.size(); ++i)
        {
            for (st j = 0; j < 6; ++j)
            {
                prop[i][j] = atof(argv[j + 2 + 6 * i]);
            };
        };

        cst approximation_number = atoi(argv[prop.size() * 6 + 2]);
        cst refine = atoi(argv[prop.size() * 6 + 3]);
        cdbl R = atof(argv[prop.size() * 6 + 4]);

        vec<str> folder(7);
        folder[0] = argv[1];
        for (st i = 0; i < prop.size(); ++i)
        {
            for (st j = 0; j < 6; ++j)
            {
                folder[i+1] += argv[j + 2 + 6 * i];
                if (j < 5) folder[i+1] += '_';
            };
        };
        folder[3] = "fiber";
        folder[4] = "circle_pixel";
        folder[5] = argv[prop.size() * 6 + 4];
        folder[6] = argv[prop.size() * 6 + 3];
        const str path = make_path(folder);

        std::cout << prop[0][0] << " " << prop[0][3]  << std::endl;
        std::cout << prop[1][0] << " " << prop[1][3]  << std::endl;
        std::cout << refine << " " << R << std::endl;

        solve_cell_elastic_problem_circle (path, approximation_number, prop, refine, R);
    };
    
    return 0;
}
#endif
