/*
 * =====================================================================================
 *
 *       Filename:  1.cpp
 *
 *    Description:  optimization
 *
 *        Version:  1.0
 *        Created:  24.07.2012 13:53:45
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
//#include <stdlib.h>
#include <stdio.h>

int main()
{
    unsigned long crc_table[256];
    unsigned long crc;

    for (int i = 0; i < 256; i++)
    {
        crc = i;
        for (int j = 0; j < 8; j++)
            crc = crc & 1 ? (crc >> 1) ^ 0xEDB88320UL : crc >> 1;

        crc_table[i] = crc;
    }

    printf("%ld",crc_table[10]);

    return 0;
}

