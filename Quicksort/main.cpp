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
        2, 0, 3, 1, 27, 2, 12, 3, 58, 4, 45, 5, 95, 6, 25, 7, 67,
        8, 31, 9, 46, 10, 76, 11, 24, 12, 74, 13, 86, 14, 19, 15
    };*/
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = 1 << 25;
    uint_t interval = 1 << 20;
    bool orderAsc = true;
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);
    fillTable(input, tableLen, interval);
    /*printTable(input, tableLen);*/

    sortParallel(input, outputParallel, tableLen, orderAsc);
    /*printTable(outputParallel, tableLen);*/

    printf("\n");
    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);

    ////cudaFreeHost(inputData);
    cudaFreeHost(outputParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
