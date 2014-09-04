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

    sortParallel(input, outputParallel, tableLen, orderAsc);

    outputCorrect = sortCorrect(input, tableLen);
    compareArrays(outputParallel, outputCorrect, tableLen);

    ////cudaFreeHost(inputData);
    //cudaFreeHost(outputDataParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
