#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/constants_common.h"
#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"
#include "data_types.h"


/*
Allocates host memory.
*/
void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **bufferSequential,
    data_t **outputCorrect, double ***timers, uint_t tableLen, uint_t testRepetitions
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
    *bufferSequential = (data_t*)malloc(tableLen * sizeof(**bufferSequential));
    checkMallocError(*bufferSequential);
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
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *bufferSequential,
    data_t *outputCorrect, double **timers
)
{
    free(input);
    free(outputParallel);
    free(outputSequential);
    free(bufferSequential);
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
void allocDeviceMemory(
    data_t **dataTable, data_t **dataBuffer, sample_t **samples, uint_t **ranksEven, uint_t **ranksOdd,
    uint_t tableLen
)
{
    cudaError_t error;
    uint_t samplesLen = tableLen / SUB_BLOCK_SIZE;

    error = cudaMalloc(dataTable, tableLen * sizeof(**dataTable));
    checkCudaError(error);
    error = cudaMalloc(dataBuffer, tableLen * sizeof(**dataBuffer));
    checkCudaError(error);

    error = cudaMalloc(samples, samplesLen * sizeof(**samples));
    checkCudaError(error);

    error = cudaMalloc(ranksEven, samplesLen * sizeof(**ranksEven));
    checkCudaError(error);
    error = cudaMalloc(ranksOdd, samplesLen * sizeof(**ranksOdd));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd
)
{
    cudaError_t error;

    error = cudaFree(dataTable);
    checkCudaError(error);
    error = cudaFree(dataBuffer);
    checkCudaError(error);

    error = cudaFree(samples);
    checkCudaError(error);

    error = cudaFree(ranksEven);
    checkCudaError(error);
    error = cudaFree(ranksOdd);
    checkCudaError(error);
}
