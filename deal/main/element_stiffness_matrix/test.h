/*
 * =====================================================================================
 *
 *       Filename:  test.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07.09.2012 16:19:02
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

template <int dim, class type>
class A 
{
    public:
        A();
        virtual type a () ;
        virtual type b ();
    protected:
        int con;
};

template <int dim, class type>
A<dim, type>::A ()
{

};

template <int dim, class type>
type A<dim, type>::a ()
{

};

template <int dim, class type>
type A<dim, type>::b ()
{
    return dim;

};
