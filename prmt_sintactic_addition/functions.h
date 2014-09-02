#ifndef FUNCTIONS_H_0VRXPZIE
#define FUNCTIONS_H_0VRXPZIE

class IntegerPow
{
    public:
        IntegerPow(cst number) : n(number) {};
        st operator() (cst degree) const
        {
            st res = 1;
            FOR(i, 0, degree)
                res *= n;
            return res;
        };
        cst n;
};

IntegerPow operator "" _pow(unsigned long long n)
{
   return IntegerPow(n);
};

#endif /* end of include guard: FUNCTIONS_H_0VRXPZIE */

