#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/kd.h>
#include <sys/ioctl.h>

using namespace std;

int main()
{
    // cout << "Hello world!\7" << endl;
    for (int i=0; i<8; i++)
    {
        printf("\7");
        usleep(500000);
    }
    return 0;
}
