#include "mmkernel.h"

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

#if 1

#define WIDTH  16
#define HEIGHT 16

__global__ void MatMulB(float * matA, float * matB, float * matC, unsigned int dim) {

    __shared__ float matA_s[WIDTH][HEIGHT], matB_s[WIDTH][HEIGHT];

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
    stride_k = (rem_k) ? 1 : 0;

    for (int j=0;j<stride_y;j++) {
        int eIdx_y =  (j * tot_y) + tid_y;
        if (eIdx_y >= dim)
            continue;

        for (int i=0;i<stride_x;i++) {
        int eIdx_x = (i * tot_x) + tid_x;
        if (eIdx_x >= dim)
            continue;

#if 0
            float temp = 0.0;
            int offA = eIdx_y * dim;
            int offB = eIdx_x * dim;
            for (int k=0;k<stride_k;k++) {
                int kIdx = (k * WIDTH) + threadIdx.x;
                matA_s[threadIdx.x][threadIdx.y] = matA[offA + kIdx];
                matB_s[threadIdx.x][threadIdx.y] = matB[offB + kIdx]; 
                __syncthreads();

               // temp += matA_s[
            }
#else
            float temp = 0.0;
            int offA = eIdx_y * dim;
            int offB = eIdx_x * dim;
            for (int k=0;k<dim;k++) {
                temp += matA[offA + k] * matB[offB + k];
            }
            matC[offA + eIdx_x] = temp; 
#endif
        }
    }
}
#endif

cudaError_t MatMul(float * matA, float * matB, float * matC, size_t dim) {

    //MatMulA <<<32, 512>>> (matA, matB, matC, dim);
#if 1
    dim3 grid(16, 16, 1);
    dim3 block(WIDTH, HEIGHT ,1);
    MatMulB <<<grid, block>>> (matA, matB, matC, dim);
#endif
    return cudaDeviceSynchronize();
}

