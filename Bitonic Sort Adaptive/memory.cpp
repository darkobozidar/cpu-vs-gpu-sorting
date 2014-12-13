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
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    double ***timers, uint_t tableLen, uint_t testRepetitions
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
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect,
    double **timers
)
{
    free(input);
    free(outputParallel);
    free(outputSequential);
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
    data_t **dataTable, data_t **dataBuffer, interval_t **d_intervals, interval_t **d_intervalsBuffer, uint_t tableLen
)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t phasesAll = log2((double)tableLenPower2);
    uint_t phasesBitonicMerge = log2((double)2 * THREADS_PER_MERGE);
    uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    cudaError_t error;

    // Table is padded to next power of 2 in length
    error = cudaMalloc(dataTable, tableLenPower2 * sizeof(**dataTable));
    checkCudaError(error);
    error = cudaMalloc(dataBuffer, tableLenPower2 * sizeof(**dataBuffer));
    checkCudaError(error);

    error = cudaMalloc(d_intervals, intervalsLen * sizeof(**d_intervals));
    checkCudaError(error);
    error = cudaMalloc(d_intervalsBuffer, intervalsLen * sizeof(**d_intervalsBuffer));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(data_t *dataTable, data_t *dataBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer)
{
    cudaError_t error;

    error = cudaFree(dataTable);
    checkCudaError(error);
    error = cudaFree(dataBuffer);
    checkCudaError(error);

    error = cudaFree(d_intervals);
    checkCudaError(error);
    error = cudaFree(d_intervalsBuffer);
    checkCudaError(error);
}
