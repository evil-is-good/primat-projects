#ifndef CLASSB_H_
#define CLASSB_H_
#include "classA.h"

class classB
{
public:
	classB();
	~classB();

protected:
	classA a;
};

classB::classB() : a ()
{
	printf("classB constructor\n");
}

classB::~classB()
{
	printf("classB destructor\n");
}

#endif
