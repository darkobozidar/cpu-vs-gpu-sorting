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
    //el_t *input;
    el_t input[32] = {
        6, 0, 23, 1, 29, 2, 35, 3, 45, 4, 63, 5, 64, 6, 97, 7, 1, 8, 4, 9, 25, 10, 34, 11,
        45, 12, 67, 13, 98, 14, 99, 15, 4, 16, 19, 17, 41, 18, 58, 19, 68, 20, 80, 21, 81,
        22, 96, 23, 4, 24, 13, 25, 18, 26, 33, 27, 55, 29, 66, 29, 88, 30, 90, 32
    };
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = 1 << 5;
    uint_t interval = 1 << 16;
    bool orderAsc = true;  // TODO use this
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    //error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    //checkCudaError(error);
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);
    //fillTable(input, tableLen, interval);
    //printTable(input, tableLen);

    sortParallel(input, outputParallel, tableLen, orderAsc);
    //printTable(outputParallel, tableLen);

    printf("\n");
    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);

    ////cudaFreeHost(inputData);
    //cudaFreeHost(outputDataParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
