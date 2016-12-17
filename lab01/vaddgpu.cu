#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"
#include "cuda.h"

#define CHECK(exp)                                          \
    do {                                                    \
        if (exp != 0) {                                     \
            printf("Runtime error at line %d\n", __LINE__); \
            exit(-1);                                       \
        }                                                   \
    } while(0)

#define CUCHECK(exp)                                             \
    do {                                                         \
        if (exp != cudaSuccess) {                                \
            printf("CUDA runtime error at line %d\n", __LINE__); \
            exit(-1);                                            \
        }                                                        \
    } while(0)

__global__ void vectorAddA(float * vectorA, float * vectorB, float * vectorC, unsigned int count) {

    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (threadId < count)
        vectorC[threadId] = vectorA[threadId] + vectorB[threadId];
}

__global__ void vectorAddB(float * vectorA, float * vectorB, float * vectorC, unsigned int count) {
    //Play with various versions of kernels (depending on threads and vice versa WBB)

    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

    int totalThreads = gridDim.x * blockDim.x;
    int stride = count / totalThreads;
    int remainder = count % totalThreads;
    stride += (remainder) ? 1 : 0;

    for (int i=0;i<stride;i++) {
        unsigned long idx = (i * totalThreads) + threadId;
        if (idx < count)
            vectorC[idx] = vectorA[idx] + vectorB[idx];
    }
}

int main(int argc, char ** argv) {

    float * veca_h, * vecb_h, * vecc_h, * vecc_hv;
    float * veca_d, * vecb_d, * vecc_d;
    size_t  count;

    //Loading vector-A and -B from file for calculation on GPU
    //Loading vector-C from file for verifying the results from GPU
    CHECK(    binReadAsArrayNP<float>("vecA.bin", NULL, &veca_h, &count));
    CHECK(    binReadAsArrayNP<float>("vecB.bin", NULL, &vecb_h, &count));
    CHECK(    binReadAsArrayNP<float>("vecC.bin", NULL, &vecc_hv, &count));
    vecc_h = new float [count];
    
    //Allocate memory on GPU for vectors (WBB)
    CUCHECK(    cudaMalloc(&veca_d, count * sizeof(float)));
    CUCHECK(    cudaMalloc(&vecb_d, count * sizeof(float)));
    CUCHECK(    cudaMalloc(&vecc_d, count * sizeof(float)));

    //Copy vector-A and -B from the host to the device (WBB)
    CUCHECK(    cudaMemcpy(veca_d, veca_h, count * sizeof(float), cudaMemcpyHostToDevice));
    CUCHECK(    cudaMemcpy(vecb_d, vecb_h, count * sizeof(float), cudaMemcpyHostToDevice));
    CUCHECK(    cudaMemset(vecc_d, 0, count * sizeof(float)));

    //Perform the vector addition by calling the kernel
    //Play with scaling and configuration (WBB)
    Timer compTimer;
    compTimer.Start();
    vectorAddB<<<128, 256>>>(veca_d, vecb_d, vecc_d, count);
    cudaDeviceSynchronize();
    compTimer.Stop();
    Timer::Duration d = compTimer.GetDuration();
    printf("Computation time: %.2f us\n", d.raw);

    //Copy vector-C which is the result back from GPU
    CUCHECK(    cudaMemcpy(vecc_h, vecc_d, count * sizeof(float), cudaMemcpyDeviceToHost));

    //Verification
    for (size_t idx=0;idx<count;idx++) {
        if (vecc_h[idx] != vecc_hv[idx]) {
            printf("Verification: FAILED (%ld)\n", idx);
            exit(-1);
        }
    }
    printf("Verification: PASSED\n");

    //Release resources after completed calculation (WBB)
    CHECK(    binDiscardArrayNP(veca_h));
    CHECK(    binDiscardArrayNP(vecb_h));
    CHECK(    binDiscardArrayNP(vecc_hv));
    CUCHECK(    cudaFree(veca_d));
    CUCHECK(    cudaFree(vecb_d));
    CUCHECK(    cudaFree(vecc_d));

    return 0;
}

