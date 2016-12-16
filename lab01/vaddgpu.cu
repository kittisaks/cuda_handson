#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"
#include "cuda.h"

int main(int argc, char ** argv) {

#if 0
    float * vectorA;
    binReadAsArray<float>("gold/vecA.bin", NULL, &vectorA);

    for (size_t idx=0;idx<1000;idx++)
        printf("%.4f\n", vectorA[idx]);
#else
    float * vectorA;
    BinInfo bi;
    if (binReadAsArrayNP<float>("gold/vecA.bin", &bi, &vectorA) == -1)
        printf("ERROR!\n");

    for (size_t idx=0;idx<1000;idx++)
        printf("%.4f\n", vectorA[idx]);

    void * memd;
    cudaMalloc(&memd, bi.size);
    cudaMemcpy(memd, vectorA, bi.size - bi.offset, cudaMemcpyHostToDevice);

    binDiscardArrayNP(vectorA);
#endif

    return 0;
}

