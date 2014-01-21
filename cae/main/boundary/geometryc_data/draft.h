template <int dim>
class GeometricDataBoundary
{
};

template<>
class GeometricDataBoundary<0>
{
    public:
        double get_coor() { return coor; };

    private:
        double coor = 0.0;
};

template<>
class GeometricDataBoundary<1>
{
};

template <int spacedim>
class Point
{
    public:
    virtual x() = 0;
    virtual y() = 0;
    virtual z() = 0;
};
