#ifndef BLACK_ON_WHITE_SUBSTITUTER_DESC_2

#define BLACK_ON_WHITE_SUBSTITUTER_DESC_2

#include <projects/prmt_sintactic_addition/prmt_sintactic_addition.h>

namespace prmt
{
    class BlackOnWhiteSubstituter
    {
        public:
            BlackOnWhiteSubstituter  ();
            BlackOnWhiteSubstituter  (cst n);
            ~BlackOnWhiteSubstituter ();

            //METHODS
        public:
            st   subst    (cst index) const;
            bool is_black (cst index) const;

        private:
            void add_white_and_black (cst w, cst b);

            //FIELDS
        public:
            vec<st> black;
            vec<st> white;
            st  size;

        public:
            template<u8 dim, u8 spacedim>
                friend class DomainLooper;
    };
};

#endif
