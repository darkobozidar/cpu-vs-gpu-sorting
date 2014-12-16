#include <stdlib.h>
#include <math.h>

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
void allocDeviceMemory(
    data_t **dataTableKeys, data_t **dataTableValues, data_t **dataBufferKeys, data_t **dataBufferValues,
    interval_t **d_intervals, interval_t **d_intervalsBuffer, uint_t tableLen
)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t phasesAll = log2((double)tableLenPower2);
    uint_t phasesBitonicMerge = log2((double)2 * THREADS_PER_MERGE);
    uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    cudaError_t error;

    // Table is padded to next power of 2 in length
    error = cudaMalloc(dataTableKeys, tableLenPower2 * sizeof(**dataTableKeys));
    checkCudaError(error);
    error = cudaMalloc(dataTableValues, tableLenPower2 * sizeof(**dataTableValues));
    checkCudaError(error);
    error = cudaMalloc(dataBufferKeys, tableLenPower2 * sizeof(**dataBufferKeys));
    checkCudaError(error);
    error = cudaMalloc(dataBufferValues, tableLenPower2 * sizeof(**dataBufferValues));
    checkCudaError(error);

    error = cudaMalloc(d_intervals, intervalsLen * sizeof(**d_intervals));
    checkCudaError(error);
    error = cudaMalloc(d_intervalsBuffer, intervalsLen * sizeof(**d_intervalsBuffer));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataTableKeys, data_t *dataTableValues, data_t *dataBufferKeys, data_t *dataBufferValues,
    interval_t *d_intervals, interval_t *d_intervalsBuffer
)
{
    cudaError_t error;

    error = cudaFree(dataTableKeys);
    checkCudaError(error);
    error = cudaFree(dataTableValues);
    checkCudaError(error);
    error = cudaFree(dataBufferKeys);
    checkCudaError(error);
    error = cudaFree(dataBufferValues);
    checkCudaError(error);

    error = cudaFree(d_intervals);
    checkCudaError(error);
    error = cudaFree(d_intervalsBuffer);
    checkCudaError(error);
}
