#ifndef TMP
#define TMP 1
#include <stdio.h>
#include <array>
#include <iostream>
#include <fstream>

int main(int argc, char const *argv[])
{
    std::array<std::array<double, 3>, 2> arr = {std::array<double, 3>{1,4,3}, std::array<double, 3>{2,5,5}};
    {
        std::ofstream out ("test.bin", std::ios::out | std::ios::binary);
        out.write ((char *) &arr, sizeof arr);
        // for (int i = 0; i < arr.size(); ++i)
        // {
        //     for (int j = 0; j < arr[i].size(); ++j)
        //     {
        //         out.write ((char *) &(arr[i][j]), sizeof(int));
        //     };
        // };
        // int a = 30;
        // out.write ((char *) &a, sizeof a);
        out.close ();
    };

	
	printf("ffff\n");
	return 0;
}
#endif
