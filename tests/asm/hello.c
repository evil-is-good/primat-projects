#include <unistd.h>
 
int main(int argc, char* argv[])
{
    char str[] = "A\n";
    //int n = 10;
        write(1, str, 1); //sizeof(argv[1]) - 1);
          _exit(0);
}
