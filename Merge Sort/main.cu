#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char** argv) {
    // Rename array to table everywhere in code
    data_t input[32] = { 6, 23, 29, 35, 45, 63, 64, 97, 1, 4, 25, 34, 45, 67, 98, 99, 4, 19, 41, 58, 68, 80, 81, 96, 4, 13, 18, 33, 55, 66, 88, 90 };;
    //data_t* input;
    data_t* outputParallel;
    data_t* outputSequential;
    data_t* outputCorrect;

    uint_t tableLen = 1 << 5;
    uint_t blockSize;
    bool orderAsc = TRUE;
    cudaError_t error;

    LARGE_INTEGER timerStart;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    // Memory allocation on device
    /*error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    checkCudaError(error);*/
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);

    //fillArrayRand(input, tableLen);
    //fillArrayValue(input, tableLen, 5);

    sortParallel(input, outputParallel, tableLen, orderAsc);
    printArray(outputParallel, tableLen);

    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);
    // TODO free memory

    // cudaFreeHost(input);
    cudaFreeHost(outputParallel);
    //free(outputSequential);
    free(outputCorrect);

    getchar();
    return 0;
}
