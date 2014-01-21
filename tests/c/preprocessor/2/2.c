#define N00L 0

#define REZ N00L 


#define INC(a) a 1
#define I(x) x 1

//#define INC
//
//#ifdef INC
//    #define BUFF REZ
//    #undef REZ
//    #define REZ BUFF 1
//    #undef BUFF
//    #undef INC
//#endif

#define double 0
#define call_with_1(x) x 1

#define REZ call_with_1(call_with_1(0))

REZ

INC( REZ )

REZ

#define a3 0
#define a2 a3 1
#undef a3
#define a1 a2 1

a1

#define abc I(I(I(I(0))))
#define buff abc
#undef abc
#define abc I(I(buff))

abc
