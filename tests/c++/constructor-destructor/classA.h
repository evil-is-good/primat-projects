#ifndef CLASSA_H_
#define CLASSA_H_

#include <stdio.h>

class classA
{
public:
	classA();
	~classA();
};

classA::classA()
{
	printf("classA constructor\n");
}

classA::~classA()
{
	printf("classA destructor\n");
}

#endif
