/*
 * =====================================================================================
 *
 *       Filename:  tensor.desc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06.09.2012 15:25:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef FEMENIST_TENSOR_DESC

#define FEMENIST_TENSOR_DESC

namespace Femenist
{
    template<class type>
    class Tensor
    {
        public:
//            Tensor ();
            Tensor (size_t size);
            ~Tensor ();

        //Methods
        public:
            Report fill (type *filler, size_t filler_size);
            void   fill (type filler);

        //Fields
        public:
            type operator() (size_t i);
            type operator() (size_t i, size_t j);
            type operator() (size_t i, size_t j, size_t k, size_t l);

        private:
            type *content;
//            dealii::Tensor<1, size, type> content;
            size_t t_size;
            bool is_2_rank;

    };

};

#endif
