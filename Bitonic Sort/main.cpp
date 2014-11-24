#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"
#include "sort_parallel.h"
#include "sort_sequential.h"


int main(int argc, char** argv) {
    el_t *input;
    el_t *outputParallel;
    el_t *outputCorrect;

    uint_t tableLen = (1 << 10);
    uint_t interval = 1 << 31;
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC
    cudaError_t error;

    cudaFree(NULL);  // Initializes CUDA, because CUDA init is lazy
    srand(time(NULL));

    error = cudaHostAlloc(&input, tableLen * sizeof(*input), cudaHostAllocDefault);
    checkCudaError(error);
    error = cudaHostAlloc(&outputParallel, tableLen * sizeof(*outputParallel), cudaHostAllocDefault);
    checkCudaError(error);

    for (uint_t i = 0; i < 10; i++) {
        fillTable(input, tableLen, interval);
        //printTable(input, tableLen);

        sortParallel(input, outputParallel, tableLen, sortOrder);
    }

    /*printTable(outputParallel, tableLen);*/

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
