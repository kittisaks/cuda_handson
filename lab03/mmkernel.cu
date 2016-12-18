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

#if 1

#define WIDTH  10
#define HEIGHT 10

__global__ void MatMulB(float * matA, float * matB, float * matC, unsigned int dim, float * test, float * test1) {

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

#if 1
            float temp = 0.0;
            int offA = eIdx_y * dim;
            int offB = eIdx_x * dim;
            for (int k=0;k<stride_k;k++) {

///*

                matA_s[threadIdx.y][threadIdx.x] = 0.0;
                matB_s[threadIdx.y][threadIdx.x] = 0.0;
                __syncthreads();

                int kIdx = (k * WIDTH) + threadIdx.x;
                if (kIdx < dim) {
                    matA_s[threadIdx.y][threadIdx.x] = matA[offA + kIdx];
                }

                if (threadIdx.y == 0) {
                    for (int l=0;l<WIDTH;l++) {
                        if ((k * WIDTH) + l < dim)
                            matB_s[threadIdx.x][l] = matB[offB + (k * WIDTH) + l]; 
                    }
                }
                __syncthreads();
//*/
                int l_max = (WIDTH > dim) ? dim : WIDTH; //Handle small matrices
                for (int l=0;l<l_max;l++)
                    //temp += matA[offA + (k * WIDTH) + l] * matB[offB + (k * WIDTH) + l];
                    temp += matA_s[threadIdx.y][l] * matB_s[threadIdx.x][l];

#if 0
                //test
                if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
                    for (int a=0;a<16;a++) {
                        for (int b=0;b<16;b++) {
                            test[(a*16)+b]  = matA_s[a][b];
                            test1[(a*16)+b] = matB_s[a][b];
                        }
                    }
                }
#endif
            __syncthreads();
            }
            matC[offA + eIdx_x] = temp;
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

#if 1
    dim3 grid(16, 16, 1);
    dim3 block(WIDTH, HEIGHT ,1);
    MatMulB <<<grid, block>>> (matA, matB, matC, dim, NULL, NULL);
#else
#define TDIM WIDTH * HEIGHT
    float * test_d, * test_h, * test1_d, * test1_h;
    test_h = new float [TDIM];
    test1_h = new float [TDIM];
    cudaMalloc((void **) &test_d, TDIM * sizeof(float));
    cudaMalloc((void **) &test1_d, TDIM * sizeof(float));
    cudaMemset(test_d, 0, TDIM * sizeof(float));
    cudaMemset(test1_d, 0, TDIM * sizeof(float));

    dim3 grid(1,1,1);
    dim3 block(WIDTH, HEIGHT ,1);
    MatMulB <<<grid, block>>> (matA, matB, matC, dim, test_d, test1_d);
    cudaDeviceSynchronize();

    cudaMemcpy(test_h, test_d, TDIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test1_h, test1_d, TDIM * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            printf("%.2f\t", test_h[(i*16)+j]);
        }
        printf("\n");
    }
    printf("\n\n");
            
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            printf("%.2f\t", test1_h[(i*16)+j]);
        }
        printf("\n");
    }
    printf("\n\n");
#endif
            
#endif
    return cudaDeviceSynchronize();
}

