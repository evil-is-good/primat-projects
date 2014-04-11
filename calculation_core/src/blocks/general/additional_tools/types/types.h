#ifndef types_def
#define types_def 1

namespace ATools
{
    using SecondOrderTensor = arr<arr<dbl, 2>, 2>;
    using FourthOrderTensor = arr<arr<arr<arr<dbl, 3>, 3>, 3>, 3>;
}

#endif
