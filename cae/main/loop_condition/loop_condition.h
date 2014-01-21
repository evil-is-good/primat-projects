
#ifndef LOOP_CONDITION

#define LOOP_CONDITION
 

#include "projects/cae/main/point/point.h"

namespace prmt
{
    template<int spacedim>
        class LoopCondition 
        {
            public:
                LoopCondition (prmt::Point<spacedim> &w, vec<prmt::Point<spacedim>> &b) :
                    substitutiv(w),     
                    substitutable(b) {};

                LoopCondition (prmt::Point<spacedim> &w, prmt::Point<spacedim> &b) :
                    substitutiv(w),     
                    substitutable(vec<prmt::Point<spacedim>>({b})) {};

                prmt::Point<spacedim> substitutiv;
                vec<prmt::Point<spacedim>> substitutable;

            private:
                LoopCondition () {};
        };
};

#endif
