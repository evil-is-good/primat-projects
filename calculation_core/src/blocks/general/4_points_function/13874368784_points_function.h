#ifndef points_function_def
#define points_function_def 1

#include <cmath>
#include </home/primat/projects/prmt_sintactic_addition/prmt_sintactic_addition.h>
#include </home/primat/projects/calculation_core/blocks/general/point/point.h>

template <u8 dim>
class Scalar4PointsFunc
{
};

template <>
class Scalar4PointsFunc<2>
{
    class LinFunc
    {
        public:
            LinFunc (cdbl _a, cdbl _b, cdbl _c);
            LinFunc () = delete;

            dbl operator() (cdbl x, cdbl y) const;
            dbl dx (cdbl x, cdbl y) const;
            dbl dy (cdbl x, cdbl y) const;

        private:
            cdbl a, b, c;
    };

    class LocVar
    {
        public:
            LocVar (cdbl _A, const LinFunc _C, const LinFunc _B);
            LocVar () = delete;

            arr<dbl,2> operator() (cdbl x, cdbl y) const;
            arr<dbl,2> dx (cdbl x, cdbl y) const;
            arr<dbl,2> dy (cdbl x, cdbl y) const;

        private:
            arr<dbl,2> quad_roots (cdbl a, cdbl b, cdbl c) const;

            cdbl A;
            const LinFunc B, C;
    };

    class LocFunc
    {
        public:
            LocFunc (arr<dbl, 4> glob_val);
            LocFunc () = delete;

        dbl operator() (cdbl s, cdbl r) const;
        dbl dx (cdbl s, cdbl r, cdbl s_dx, cdbl r_dx) const;
        dbl dy (cdbl s, cdbl r, cdbl s_dy, cdbl r_dy) const;

        private:
            cdbl a, b, c, d;
    };

    public:
        Scalar4PointsFunc (arr<prmt::Point<2>, 4> nodes,
                           arr<dbl, 4> f_values);
        Scalar4PointsFunc () = delete;

        dbl operator() (cdbl x, cdbl y) const;
        dbl dx (cdbl x, cdbl y) const;
        dbl dy (cdbl x, cdbl y) const;

    private:
        cdbl a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y;
        const LocVar s, r;
        const LocFunc f_;

};

/////////////////////////////Scalar4PointsFunc
Scalar4PointsFunc<2>::Scalar4PointsFunc (arr<prmt::Point<2>, 4> nodes,
                                         arr<dbl, 4> f_values) :
    a_x (nodes[0].x()),
    b_x (nodes[1].x() - nodes[0].x()),
    c_x (nodes[3].x() - nodes[0].x()),
    d_x ((nodes[0].x() + nodes[2].x()) - (nodes[1].x() + nodes[3].x())),
    a_y (nodes[0].y()),
    b_y (nodes[1].y() - nodes[0].y()),
    c_y (nodes[3].y() - nodes[0].y()),
    d_y ((nodes[0].y() + nodes[2].y()) - (nodes[1].y() + nodes[3].y())),

    s (
            b_y * d_x - b_x * d_y, 
            LinFunc(a_x * d_y - a_y * d_x - b_y * c_x + b_x * c_y, -d_y,  d_x), 
            LinFunc(a_y * c_x - a_x * c_y,                          c_y, -c_x)
      ),

    r (
            c_y * d_x - c_x * d_y, 
            LinFunc(a_x * d_y - a_y * d_x + b_y * c_x - b_x * c_y, -d_y,  d_x), 
            LinFunc(a_y * b_x - a_x * b_y,                          b_y, -b_x)
      ),

    f_(f_values)
{};

dbl Scalar4PointsFunc<2>::operator() (cdbl x, cdbl y) const
{
    return f_(s(x, y)[0], r(x, y)[1]);
};

dbl Scalar4PointsFunc<2>::dx (cdbl x, cdbl y) const
{
    return f_.dx(s(x, y)[0], r(x, y)[1], s.dx(x, y)[0], r.dx(x, y)[1]);
};

dbl Scalar4PointsFunc<2>::dy (cdbl x, cdbl y) const
{
    return f_.dx(s(x, y)[0], r(x, y)[1], s.dy(x, y)[1], r.dy(x, y)[0]);
};
///////////////////////////

//////////////////////////LocFunc
Scalar4PointsFunc<2>::LocFunc::LocFunc (arr<dbl, 4> glob_val) :
    a (glob_val[0]),
    b (glob_val[1] - glob_val[0]),
    c (glob_val[3] - glob_val[0]),
    d ((glob_val[0] + glob_val[2]) - (glob_val[1] + glob_val[3]))
{};

dbl Scalar4PointsFunc<2>::LocFunc::operator() (cdbl s, cdbl r) const
{
    return a + b * s + c * r + d * s * r;
};

dbl Scalar4PointsFunc<2>::LocFunc::dx (cdbl s, cdbl r, cdbl s_dx, cdbl r_dx) const
{
    return b * s_dx + c * r_dx + d * (s_dx * r + s * r_dx);
};

dbl Scalar4PointsFunc<2>::LocFunc::dy (cdbl s, cdbl r, cdbl s_dy, cdbl r_dy) const
{
    return b * s_dy + c * r_dy + d * (s_dy * r + s * r_dy);
};
/////////////////////////

///////////////////////////LocVar
Scalar4PointsFunc<2>::LocVar::LocVar (cdbl _A, const LinFunc _C, const LinFunc _B) :
    A (_A),
    B (_B),
    C (_C)
{};

arr<dbl,2> Scalar4PointsFunc<2>::LocVar::operator() (cdbl x, cdbl y) const
{
    if (std::abs(A) < 1e-10)
    {
        return quad_roots(A, B(x, y), C(x, y));
    }
    else
    {
        dbl res = C(x, y) / B(x, y);
        return arr<dbl, 2>{res, res};
    };
};

arr<dbl,2> Scalar4PointsFunc<2>::LocVar::dx (cdbl x, cdbl y) const
{
    if (std::abs(A) < 1e-10)
    {
        dbl factor = 1.0 / (std::sqrt(B(x, y) * B(x, y) - 4.0 * A * C(x, y)));
        return arr<dbl, 2>{
            (B.dx(x, y) * (1.0 - B(x, y)) / (2.0 * A) - C.dx(x, y)) * factor, 
            (B.dx(x, y) * (1.0 + B(x, y)) / (2.0 * A) - C.dx(x, y)) * factor};
    }
    else
    {
        dbl res = (C.dx(x, y) * B(x, y) -  C(x, y) * B.dx(x, y)) / (B(x, y) * B(x, y));
        return arr<dbl, 2>{res, res};
    };
};

arr<dbl,2> Scalar4PointsFunc<2>::LocVar::dy (cdbl x, cdbl y) const
{
    if (std::abs(A) < 1e-10)
    {
        dbl factor = 1.0 / (std::sqrt(B(x, y) * B(x, y) - 4.0 * A * C(x, y)));
        return arr<dbl, 2>{
            (B.dy(x, y) * (1.0 - B(x, y)) / (2.0 * A) - C.dy(x, y)) * factor, 
            (B.dy(x, y) * (1.0 + B(x, y)) / (2.0 * A) - C.dy(x, y)) * factor};
    }
    else
    {
        dbl res = (C.dy(x, y) * B(x, y) -  C(x, y) * B.dy(x, y)) / (B(x, y) * B(x, y));
        return arr<dbl, 2>{res, res};
    };
};

arr<dbl,2> Scalar4PointsFunc<2>::LocVar::quad_roots (cdbl a, cdbl b, cdbl c) const
{
    dbl sq_root = std::sqrt(B(x, y) * B(x, y) - 4.0 * A * C(x, y));
    return arr<dbl, 2>{
        (B(x, y) - sq_root) / (2.0 * A),
        (B(x, y) + sq_root) / (2.0 * A)};
};
//////////////////////////

//////////////////////////LinFunc
Scalar4PointsFunc<2>::LinFunc::LinFunc (cdbl _a, cdbl _b, cdbl _c) :
    a (_a),
    b (_b),
    c (_c)
{};

dbl Scalar4PointsFunc<2>::LinFunc::operator() (cdbl x, cdbl y) const
{
    
    return a + b * x + c * y;
};

dbl Scalar4PointsFunc<2>::LinFunc::dx (cdbl x, cdbl y) const
{
    return b;
};

dbl Scalar4PointsFunc<2>::LinFunc::dy (cdbl x, cdbl y) const
{
    return c;
};
/////////////////////////


template<u8 dim>
dbl scalar_4_points_func (arr<prmt::Point<dim>,4> nodes,
                          arr<dbl, 4> f,
                          prmt::Point<dim> p)
{
};

template<>
dbl scalar_4_points_func<2> (arr<prmt::Point<2>,4> nodes,
                             arr<dbl, 4> f,
                             prmt::Point<2> p)
{
    cdbl ax = nodes[0].x();
    cdbl bx = nodes[1].x() - nodes[0].x();
    cdbl cx = nodes[3].x() - nodes[0].x();
    cdbl dx = (nodes[0].x() + nodes[2].x()) - (nodes[1].x() + nodes[3].x());

    cdbl ay = nodes[0].y();
    cdbl by = nodes[1].y() - nodes[0].y();
    cdbl cy = nodes[3].y() - nodes[0].y();
    cdbl dy = (nodes[0].y() + nodes[2].y()) - (nodes[1].y() + nodes[3].y());

    auto f_ = [f] (cdbl s, cdbl r){
        cdbl af = f[0];
        cdbl bf = f[1] - f[0];
        cdbl cf = f[3] - f[0];
        cdbl df = (f[0] + f[2]) - (f[1] + f[3]);
        return af + bf * s + cf * r + df * s * r;
    };

    enum {s, r};

    dbl A[2] = {by * dx - bx * dy,
                cy * dx - cx * dy};

    dbl B[2] = {ax * dy - ay * dx - by * cx + bx * cy - dy * p.x() + dx * p.y(),  
                ax * dy - ay * dx + by * cx - bx * cy - dy * p.x() + dx * p.y()};

    dbl C[2] = {ay * cx - ax * cy + cy * p.x() - cx * p.y(),
                ay * bx - ax * by + by * p.x() - bx * p.y()};

    auto qad_roots = [A, B, C] (cst var, cdbl x, cdbl y){
        return arr<dbl,2> {
                (B[var] - std::sqrt(B[var] * B[var] - 4.0 * A[var] * C[var])) / 
                    (2.0 * A[var]),
                (B[var] + std::sqrt(B[var] * B[var] - 4.0 * A[var] * C[var])) / 
                    (2.0 * A[var])};
    };

    auto loc_var = [=] (cst var, cdbl x, cdbl y){
        if (abs(A[var]) > 1e-10) 
            return qad_roots (var, x, y)[var];
        else
            return C[var] / B[var];
    };

    return f_(loc_var(s, p.x(), p.y()), loc_var(r, p.x(), p.y()));
};

// template<u8 dim>
// arr<dbl, 2> scalar_4_points_func_grad (arr<prmt::Point<2>,4> nodes,
//                                        arr<dbl, 4> f,
//                                        prmt::Point<2> p)
// {
// };
// 
// template<>
// arr<dbl, 2> scalar_4_points_func_grad<2> (arr<prmt::Point<2>,4> nodes,
//                                           arr<dbl, 4> f,
//                                           prmt::Point<2> p)
// {
//     cdbl ax = nodes[0].x();
//     cdbl bx = nodes[1].x() - nodes[0].x();
//     cdbl cx = nodes[3].x() - nodes[0].x();
//     cdbl dx = (nodes[0].x() + nodes[2].x()) - (nodes[1].x() + nodes[3].x());
// 
//     cdbl ay = nodes[0].y();
//     cdbl by = nodes[1].y() - nodes[0].y();
//     cdbl cy = nodes[3].y() - nodes[0].y();
//     cdbl dy = (nodes[0].y() + nodes[2].y()) - (nodes[1].y() + nodes[3].y());
// 
//     auto f_ = [f] (cdbl s_dn, cdbl r_dn){
//         cdbl af = f[0];
//         cdbl bf = f[1] - f[0];
//         cdbl cf = f[3] - f[0];
//         cdbl df = (f[0] + f[2]) - (f[1] + f[3]);
//         return af + bf * s_dn + cf * r_dn + df * (s_dn * r + s * r_dn);
//     };
// 
//     enum {s, r};
// 
//     dbl A[2] = {by * dx - bx * dy,
//                 cy * dx - cx * dy};
// 
//     dbl B[2] = {ax * dy - ay * dx - by * cx + bx * cy - dy * p.x() + dx * p.y(),  
//                 ax * dy - ay * dx + by * cx - bx * cy - dy * p.x() + dx * p.y()};
// 
//     dbl C[2] = {ay * cx - ax * cy + cy * p.x() - cx * p.y(),
//                 ay * bx - ax * by + by * p.x() - bx * p.y()};
// 
//     auto qad_roots = [A, B, C] (cst var, cdbl x, cdbl y){
//         return arr<dbl,2> {
//                 (B[var] - std::sqrt(B[var] * B[var] - 4.0 * A[var] * C[var])) / 
//                     (2.0 * A[var]),
//                 (B[var] + std::sqrt(B[var] * B[var] - 4.0 * A[var] * C[var])) / 
//                     (2.0 * A[var])};
//     };
// 
//     auto loc_var = [=] (cst var, cdbl x, cdbl y){
//         if (abs(A[var]) > 1e-10) 
//             return qad_roots (var, x, y)[var];
//         else
//             return C[var] / B[var];
//     };
// 
//     return f_(loc_var(s, p.x(), p.y()), loc_var(r, p.x(), p.y()));
// };

#endif
