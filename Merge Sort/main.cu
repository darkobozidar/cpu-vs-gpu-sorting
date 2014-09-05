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
    el_t *input;
    /*el_t input[32] = {
        7, 0, 19, 1, 20, 2, 31, 3, 48, 4, 49, 5, 54, 6, 68, 7, 2, 8, 14, 9, 41, 10, 46,
        11, 60, 12, 62, 13, 63, 14, 96, 15, 12, 16, 17, 17, 37, 18, 40, 19, 64, 20, 66,
        21, 88, 22, 97, 23, 24, 24, 26, 25, 37, 26, 75, 27, 76, 28, 76, 29, 76, 30, 85, 31
    };*/
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = 1 << 18;
    uint_t interval = 1 << 16;
    bool orderAsc = true;  // TODO use this
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
