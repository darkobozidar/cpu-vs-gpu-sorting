#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"


/*
Allocates host memory.
*/
void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **outputSequentialKeys, data_t **outputSequentialValues, data_t **outputCorrect, double ***timers,
    uint_t tableLen, uint_t testRepetitions
)
{
    // Data input
    *inputKeys = (data_t*)malloc(tableLen * sizeof(**inputKeys));
    checkMallocError(*inputKeys);
    *inputValues = (data_t*)malloc(tableLen * sizeof(**inputValues));
    checkMallocError(*inputValues);

    // Data output
    *outputParallelKeys = (data_t*)malloc(tableLen * sizeof(**outputParallelKeys));
    checkMallocError(*outputParallelKeys);
    *outputParallelValues = (data_t*)malloc(tableLen * sizeof(**outputParallelValues));
    checkMallocError(*outputParallelValues);
    *outputSequentialKeys = (data_t*)malloc(tableLen * sizeof(**outputSequentialKeys));
    checkMallocError(*outputSequentialKeys);
    *outputSequentialValues = (data_t*)malloc(tableLen * sizeof(**outputSequentialValues));
    checkMallocError(*outputSequentialValues);
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);

    // Stopwatch times for PARALLEL, SEQUENTIAL and CORREECT
    double** timersTemp = new double*[NUM_STOPWATCHES];
    for (uint_t i = 0; i < NUM_STOPWATCHES; i++)
    {
        timersTemp[i] = new double[testRepetitions];
    }

    *timers = timersTemp;
}

/*
Frees host memory.
*/
void freeHostMemory(
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *outputSequentialKeys, data_t *outputSequentialValues, data_t *outputCorrect, double **timers
)
{
    free(inputKeys);
    free(inputValues);
    free(outputParallelKeys);
    free(outputParallelValues);
    free(outputSequentialKeys);
    free(outputSequentialValues);
    free(outputCorrect);

    for (uint_t i = 0; i < NUM_STOPWATCHES; ++i)
    {
        delete[] timers[i];
    }
    delete[] timers;
}

/*
Allocates device memory.
*/
void allocDeviceMemory(data_t **dataTableKeys, data_t **dataTableValues, uint_t tableLen)
{
    cudaError_t error;

    error = cudaMalloc(dataTableKeys, tableLen * sizeof(**dataTableKeys));
    checkCudaError(error);
    error = cudaMalloc(dataTableValues, tableLen * sizeof(**dataTableValues));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(data_t *dataTableKeys, data_t *dataTableValues)
{
    cudaError_t error;

    error = cudaFree(dataTableKeys);
    checkCudaError(error);
    error = cudaFree(dataTableValues);
    checkCudaError(error);
}
