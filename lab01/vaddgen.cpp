#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"

void InitRandomNumberGenerator() {

    srand(100);
}

int GenerateVectorFloat(float * vec, size_t size) {

    for (size_t idx=0;idx<size;idx++) {
        vec[idx] = ((float) rand() / (float) RAND_MAX) * 100;
    }

    return 0;
}

int VectorAddFloat(float * vecC, float * vecA, float * vecB, size_t size) {

    for (size_t idx=0;idx<size;idx++) 
        vecC[idx] = vecA[idx] + vecB[idx];

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

    size_t count = atoi(argv[1]);
    if (count <= 0) {
        printf("Invalid size of vector\n");
    }

    InitRandomNumberGenerator();

    float * vectorA = new float[count];
    float * vectorB = new float[count];
    float * vectorC = new float[count];

    TIMER_PERIOD(genA_timer, GenerateVectorFloat(vectorA, count));
    TIMER_PERIOD(genB_timer, GenerateVectorFloat(vectorB, count));
    TIMER_PERIOD(add_timer, VectorAddFloat(vectorC, vectorA, vectorB, count));

    binWriteArray<float>("vecA.bin", NULL, vectorA, count);
    binWriteArray<float>("vecB.bin", NULL, vectorB, count);
    binWriteArray<float>("vecC.bin", NULL, vectorC, count);

    Timer::Duration genA_duration = genA_timer.GetDuration();
    Timer::Duration genB_duration = genB_timer.GetDuration();
    Timer::Duration add_duration = add_timer.GetDuration();

    printf("%.4f\n", genA_duration.raw);
    printf("%.4f\n", genB_duration.raw);
    printf("%.4f\n", add_duration.raw);

    return 0;
}

