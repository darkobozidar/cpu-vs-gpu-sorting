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
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    uint_t **globalBucketOffsets, double ***timers, uint_t tableLen, uint_t testRepetitions
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

    // Offsets of all global buckets
    *globalBucketOffsets = (uint_t*)malloc((NUM_SAMPLES + 1) * sizeof(**globalBucketOffsets));
    checkMallocError(*globalBucketOffsets);

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
    uint_t *globalBucketOffsets, double **timers
)
{
    free(input);
    free(outputParallel);
    free(outputSequential);
    free(outputCorrect);
    free(globalBucketOffsets);

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
    data_t **dataTable, data_t **dataBuffer, data_t **samples, uint_t **localBucketSizes,
    uint_t **localBucketOffsets, uint_t **globalBucketOffsets, uint_t tableLen
)
{
    uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    // If table length is not multiple of number of elements processed by one thread block in initial
    // bitonic sort, than array is padded to that length.
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerInitBitonicSort);
    uint_t localSamplesDistance = (elemsPerInitBitonicSort - 1) / NUM_SAMPLES + 1;
    uint_t localSamplesLen = (tableLenRoundedUp - 1) / localSamplesDistance + 1;
    // (number of all data blocks (tiles)) * (number buckets generated from NUM_SAMPLES)
    uint_t localBucketsLen = ((tableLenRoundedUp - 1) / elemsPerInitBitonicSort + 1) * (NUM_SAMPLES + 1);
    cudaError_t error;

    error = cudaMalloc(dataTable, tableLenRoundedUp * sizeof(**dataTable));
    checkCudaError(error);
    error = cudaMalloc(dataBuffer, tableLenRoundedUp * sizeof(**dataBuffer));
    checkCudaError(error);

    error = cudaMalloc(samples, localSamplesLen * sizeof(**samples));
    checkCudaError(error);

    error = cudaMalloc(localBucketSizes, localBucketsLen * sizeof(**localBucketSizes));
    checkCudaError(error);
    error = cudaMalloc(localBucketOffsets, localBucketsLen * sizeof(**localBucketOffsets));
    checkCudaError(error);
    error = cudaMalloc(globalBucketOffsets, (NUM_SAMPLES + 1) * sizeof(**globalBucketOffsets));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, data_t *samples, uint_t *localBucketSizes, uint_t *localBucketOffsets,
    uint_t *globalBucketOffsets
)
{
    cudaError_t error;

    error = cudaFree(dataTable);
    checkCudaError(error);
    error = cudaFree(dataBuffer);
    checkCudaError(error);

    error = cudaFree(samples);
    checkCudaError(error);

    error = cudaFree(localBucketSizes);
    checkCudaError(error);
    error = cudaFree(localBucketOffsets);
    checkCudaError(error);
    error = cudaFree(globalBucketOffsets);
    checkCudaError(error);
}
