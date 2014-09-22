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
    //el_t *input;
    el_t input[16] = {
        2, 0, 3, 1, 5, 2, 7, 3, 8, 4, 10, 5, 13, 6, 15, 7, 17,
        8, 14, 9, 12, 10, 11, 11, 9, 12, 7, 13, 3, 14, 1, 15
    };
    cudaDeviceReset();
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = 1 << 4;
    uint_t interval = 100;
    bool orderAsc = true;
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    /*error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    checkCudaError(error);*/
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);
    //fillTable(input, tableLen, interval);
    //printTable(input, tableLen);

    sortParallel(input, outputParallel, tableLen, orderAsc);
    //printTable(outputParallel, tableLen);

    printf("\n");
    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);

    cudaFreeHost(input);
    cudaFreeHost(outputParallel );
    free(outputCorrect);

    cudaDeviceReset();
    getchar();
    return 0;
}
