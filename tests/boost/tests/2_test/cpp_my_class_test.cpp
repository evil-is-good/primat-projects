#define BOOST_TEST_MODULE my_test
#include <boost/test/included/unit_test.hpp>

#include "hpp_my_class.hpp"

bool is_even( int i )        { return i%2 == 0;  }

BOOST_AUTO_TEST_CASE( test1 )
{
    MyC myc;
    BOOST_TEST_MESSAGE ("Hello!");
    BOOST_TEST_CHECKPOINT("Again Hello");
    BOOST_CHECK_EQUAL (myc.get(), 0);

//    int *a = new int[10];
//    a[0] = 1;
//    BOOST_CHECK_EQUAL(a[0],1);

    BOOST_CHECK_PREDICATE( is_even, (15) );
    BOOST_CHECK(sizeof(int));
    printf("AAA = %ld\n",sizeof(char));
    printf("AAA = %ld\n",sizeof(short));
    printf("AAA = %ld\n",sizeof(int));
    printf("AAA = %ld\n",sizeof(long int));

}
