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
    el_t *input;
    /*el_t input[16] = {
        80, 0, 17, 1, 29, 2, 10, 3, 87, 4, 21, 5, 14, 6, 35, 7, 99,
        8, 40, 9, 84, 10, 63, 11, 79, 12, 61, 13, 8, 14, 23, 15
    };*/
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = 1 << 18;
    uint_t interval = 1 << 16;
    bool orderAsc = true;
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);
    fillTable(input, tableLen, interval);
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
