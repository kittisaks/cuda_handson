#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"

void InitRandomNumberGenerator() {

    srand(100);
}

int GenerateMatrixFloat(float * vec, size_t dim) {

    size_t count = dim * dim;
    for (size_t idx=0;idx<count;idx++) {
        vec[idx] = ((float) rand() / (float) RAND_MAX) * 100;
    }

    return 0;
}

int MatmulFloat(float * matC, float * matA, float * matB, size_t dim) {

    #pragma omp parallel for
    for (size_t iIdx=0;iIdx<dim;iIdx++) {
        size_t i = iIdx * dim;

        for (size_t jIdx=0;jIdx<dim;jIdx++) {
            size_t j = jIdx * dim;

            float temp = 0.0;
            for (size_t kIdx=0;kIdx<dim;kIdx++) {
                temp += matA[i + kIdx] * matB[j + kIdx];
                matC[(iIdx * dim) + jIdx] = temp;
            }
        }
    }

    return 0;
}

Timer genA_timer, genB_timer, add_timer;
#define TIMER_PERIOD(timer, exp) \
    timer.Start();               \
    exp;                         \
    timer.Stop()

int main(int argc, char ** argv) {

    if (argc < 2) {
        printf("Please specify the size of vector\n");
        exit(-1);
    }

    size_t dim = atoi(argv[1]);
    if (dim <= 0) {
        printf("Invalid size of vector\n");
    }

    InitRandomNumberGenerator();
    omp_set_num_threads(20);

    size_t count = dim * dim;
    float * vectorA = new float[count];
    float * vectorB = new float[count];
    float * vectorC = new float[count];

    memset(vectorC, 0, count * sizeof(float));
    TIMER_PERIOD(genA_timer, GenerateMatrixFloat(vectorA, dim));
    TIMER_PERIOD(genB_timer, GenerateMatrixFloat(vectorB, dim));
    TIMER_PERIOD(add_timer, MatmulFloat(vectorC, vectorA, vectorB, dim));

    char vafn[128], vbfn[128], vcfn[128];
    sprintf(vafn, "vecA_%ld.bin", dim);
    sprintf(vbfn, "vecB_%ld.bin", dim);
    sprintf(vcfn, "vecC_%ld.bin", dim);
    binWriteArray<float>(vafn, NULL, vectorA, count);
    binWriteArray<float>(vbfn, NULL, vectorB, count);
    binWriteArray<float>(vcfn, NULL, vectorC, count);

    Timer::Duration genA_duration = genA_timer.GetDuration();
    Timer::Duration genB_duration = genB_timer.GetDuration();
    Timer::Duration add_duration = add_timer.GetDuration();

#if 0
    for (size_t i=0;i<dim;i++) {
        for (size_t j=0;j<dim;j++) {
            printf("%.2f\t", vectorA[(i * dim) + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    for (size_t i=0;i<dim;i++) {
        for (size_t j=0;j<dim;j++) {
            printf("%.2f\t", vectorB[(i * dim) + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    for (size_t i=0;i<dim;i++) {
        for (size_t j=0;j<dim;j++) {
            printf("%.2f\t", vectorC[(i * dim) + j]);
        }
        printf("\n");
    }
    printf("\n\n");
#endif

    printf("%.4f\n", genA_duration.raw);
    printf("%.4f\n", genB_duration.raw);
    printf("%.4f\n", add_duration.raw);

    return 0;
}

