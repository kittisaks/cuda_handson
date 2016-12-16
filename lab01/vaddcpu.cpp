#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"

int main(int argc, char ** argv) {

#if 0
    float * vectorA;
    binReadAsArray<float>("gold/vecA.bin", NULL, &vectorA);

    for (size_t idx=0;idx<1000;idx++)
        printf("%.4f\n", vectorA[idx]);
#else
    size_t count;
    float * vectorA;
    BinInfo bi;
    if (binReadAsArrayNP<float>("gold/vecA.bin", &bi, &vectorA, &count) == -1)
        printf("ERROR!\n");

    for (size_t idx=0;idx<1000;idx++)
        printf("%.4f\n", vectorA[idx]);

#endif

    return 0;
}

