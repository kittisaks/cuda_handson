#include "mmkernel.h"
#include <stdio.h>

__global__ void MatMulA(float * matA, float * matB, float * matC, unsigned int dim) {

    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

    int totalThreads = gridDim.x * blockDim.x;
    int totalElements = dim * dim;
    int stride = totalElements / totalThreads;
    int remainder = totalElements % totalThreads;
    stride += (remainder) ? 1 : 0;

    for (int strd=0;strd<stride;strd++) {
        int eIdx = (strd * totalThreads) + threadId;
        if (eIdx >= totalElements)
            continue;
#if 0
        int rowA = eIdx / dim;
        int offA = rowA * dim;
        int rowB = eIdx % dim;
        int offB = rowB * dim;
#else
        int rem   = eIdx % dim;
        int offA  = eIdx - rem;
        int offB  =  rem * dim;
#endif

        float temp = 0.0;
        for (int i=0;i<dim;i++) {
            temp += matA[offA + i] * matB[offB + i];
        }
        matC[eIdx] = temp;
    }
}


#define WIDTH  25
#define HEIGHT 25

__global__ void MatMulB(float * matA, float * matB, float * matC, unsigned int dim) {

    int tid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int tot_x = gridDim.x * blockDim.x;
    int tot_y = gridDim.y * blockDim.y;
    int rem_x = dim % tot_x;
    int rem_y = dim % tot_y;
    int stride_x = dim / tot_x;
    stride_x += (rem_x) ? 1 : 0;
    int stride_y = dim / tot_y;
    stride_y += (rem_y) ? 1 : 0;

    int stride_k = dim / WIDTH;
    int rem_k = dim % WIDTH;
    stride_k += (rem_k) ? 1 : 0;

    for (int j=0;j<stride_y;j++) {
        int eIdx_y =  (j * tot_y) + tid_y;
        if (eIdx_y >= dim)
            continue;

        for (int i=0;i<stride_x;i++) {
        int eIdx_x = (i * tot_x) + tid_x;
        if (eIdx_x >= dim)
            continue;

            float temp = 0.0;
            int offA = eIdx_y * dim;
            int offB = eIdx_x * dim;
            for (int k=0;k<dim;k++)
                temp += matA[offA + k] * matB[offB + k];
            matC[offA + eIdx_x] = temp; 
            __syncthreads();
        }
    }
}

__global__ void MatMulC(float * matA, float * matB, float * matC, unsigned int dim) {

    __shared__ float matA_s[HEIGHT][WIDTH], matB_s[HEIGHT][WIDTH];

    int tid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int tot_x = gridDim.x * blockDim.x;
    int tot_y = gridDim.y * blockDim.y;
    int rem_x = dim % tot_x;
    int rem_y = dim % tot_y;
    int stride_x = dim / tot_x;
    stride_x += (rem_x) ? 1 : 0;
    int stride_y = dim / tot_y;
    stride_y += (rem_y) ? 1 : 0;

    int stride_k = dim / WIDTH;
    int rem_k = dim % WIDTH;
    stride_k += (rem_k) ? 1 : 0;

    for (int j=0;j<stride_y;j++) {
        int eIdx_y =  (j * tot_y) + tid_y;
        if (eIdx_y >= dim)
            continue;

        for (int i=0;i<stride_x;i++) {
        int eIdx_x = (i * tot_x) + tid_x;
        if (eIdx_x >= dim)
            continue;

            float temp = 0.0;
            int offA = eIdx_y * dim;
            int offB = eIdx_x * dim;
            for (int k=0;k<stride_k;k++) {

                //Reset the shared memory space
                matA_s[threadIdx.y][threadIdx.x] = 0.0;
                matB_s[threadIdx.y][threadIdx.x] = 0.0;
                __syncthreads();

                //Loading mat_A (global) to shared memory
                int kIdx = (k * WIDTH) + threadIdx.x;
                if (kIdx < dim)
                    matA_s[threadIdx.y][threadIdx.x] = matA[offA + kIdx];

                //Loading mat_B (global) to shared memory
                if (threadIdx.y == 0) {
                    kIdx = k * WIDTH;
                    for (int l=0;l<WIDTH;l++) {
                        if ((k * WIDTH) + l < dim)
                            matB_s[threadIdx.x][l] = matB[offB + kIdx + l]; 
                    }
                }
                __syncthreads();

                //Strided row-wise multiplication
                int l_max = (WIDTH > dim) ? dim : WIDTH; //Handle small matrices
                for (int l=0;l<l_max;l++)
                    temp += matA_s[threadIdx.y][l] * matB_s[threadIdx.x][l];
                __syncthreads();
            }
            matC[offA + eIdx_x] = temp;
        }
    }
}

cudaError_t MatMul(float * matA, float * matB, float * matC, size_t dim) {

    //MatMulA <<<32, 512>>> (matA, matB, matC, dim);

    dim3 grid(16, 16, 1);
    dim3 block(WIDTH, HEIGHT ,1);
    MatMulC <<<grid, block>>> (matA, matB, matC, dim);
            
    return cudaDeviceSynchronize();
}

