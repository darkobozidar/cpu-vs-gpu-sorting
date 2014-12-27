#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/constants_common.h"
#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"


/*
Allocates host memory.
*/
void allocHostMemory(
    data_t **inputKeys, data_t **inputValues, data_t **outputParallelKeys, data_t **outputParallelValues,
    data_t **outputSequentialKeys, data_t **outputSequentialValues, data_t **bufferSequentialKeys,
    data_t **bufferSequentialValues, data_t **outputCorrect, double ***timers, uint_t tableLen, uint_t testRepetitions
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
    *bufferSequentialKeys = (data_t*)malloc(tableLen * sizeof(**bufferSequentialKeys));
    checkMallocError(*bufferSequentialKeys);
    *bufferSequentialValues = (data_t*)malloc(tableLen * sizeof(**bufferSequentialValues));
    checkMallocError(*bufferSequentialValues);
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
    data_t *outputSequentialKeys, data_t *outputSequentialValues, data_t *bufferSequentialKeys,
    data_t *bufferSequentialValues, data_t *outputCorrect, double **timers
)
{
    free(inputKeys);
    free(inputValues);
    free(outputParallelKeys);
    free(outputParallelValues);
    free(outputSequentialKeys);
    free(outputSequentialValues);
    free(bufferSequentialKeys);
    free(bufferSequentialValues);
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
    data_t **dataKeys, data_t **dataValues, data_t **bufferKeys, data_t **bufferValues, uint_t **ranksEven,
    uint_t **ranksOdd, uint_t tableLen
)
{
    cudaError_t error;
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t ranksLen = (tableLenPower2 - 1) / SUB_BLOCK_SIZE + 1;

    error = cudaMalloc(dataKeys, tableLenPower2 * sizeof(**dataKeys));
    checkCudaError(error);
    error = cudaMalloc(dataValues, tableLenPower2 * sizeof(**dataValues));
    checkCudaError(error);
    error = cudaMalloc(bufferKeys, tableLenPower2 * sizeof(**bufferKeys));
    checkCudaError(error);
    error = cudaMalloc(bufferValues, tableLenPower2 * sizeof(**bufferValues));
    checkCudaError(error);

    error = cudaMalloc(ranksEven, ranksLen * sizeof(**ranksEven));
    checkCudaError(error);
    error = cudaMalloc(ranksOdd, ranksLen * sizeof(**ranksOdd));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, uint_t *ranksEven,
    uint_t *ranksOdd
)
{
    cudaError_t error;

    error = cudaFree(dataKeys);
    checkCudaError(error);
    error = cudaFree(dataValues);
    checkCudaError(error);
    error = cudaFree(bufferKeys);
    checkCudaError(error);
    error = cudaFree(bufferValues);
    checkCudaError(error);

    error = cudaFree(ranksEven);
    checkCudaError(error);
    error = cudaFree(ranksOdd);
    checkCudaError(error);
}
