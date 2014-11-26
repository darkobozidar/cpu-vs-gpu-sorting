#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"


/*
Allocates host memory.
*/
void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect, uint_t tableLen
)
{
    // Data input
    *input = (data_t*)malloc(tableLen * sizeof(**input));
    checkMallocError(*input);

    // Data output
    *outputParallel = (data_t*)malloc(tableLen * sizeof(**outputParallel));
    checkMallocError(*outputParallel);
    *outputSequential = (data_t*)malloc(tableLen * sizeof(**outputSequential));
    checkMallocError(*outputSequential);
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);
}

/*
Frees host memory.
*/
void freeHostMemory(data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect) {
    free(input);
    free(outputParallel);
    free(outputSequential);
    free(outputCorrect);
}

/*
Allocates device memory.
*/
void allocDeviceMemory(data_t **dataTable, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(dataTable, tableLen * sizeof(**dataTable));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(data_t *dataTable) {
    cudaError_t error;

    error = cudaFree(dataTable);
    checkCudaError(error);
}
