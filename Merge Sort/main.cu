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
    el_t *h_input, *h_output;

    uint_t tableLen = 1 << 3;
    uint_t interval = 5;
    bool orderAsc = true;  // TODO use this
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&h_input, tableLen * sizeof(*h_input), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&h_output, tableLen * sizeof(*h_output), cudaHostAllocDefault);
    checkCudaError(error);
    fillTable(h_input, tableLen, interval);

    sortParallel(h_input, h_output, tableLen, orderAsc);
    printTable(h_output, tableLen);

    //outputDataCorrect = sortCorrect(inputData, dataLen);
    //compareArrays(outputDataParallel, outputDataCorrect, dataLen);

    ////cudaFreeHost(inputData);
    //cudaFreeHost(outputDataParallel);
    ////free(outputDataSequential);
    //free(outputDataCorrect);

    getchar();
    return 0;
}
