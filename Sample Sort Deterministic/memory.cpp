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
    data_t **input, data_t **outputParallel, data_t **inputSequential, data_t **bufferSequential,
    data_t **outputSequential, data_t **outputCorrect, data_t **samples, uint_t **bucketSizes,
    uint_t **elementBuckets, uint_t **globalBucketOffsets, double ***timers, uint_t tableLen, uint_t testRepetitions
)
{
    // Data input
    *input = (data_t*)malloc(tableLen * sizeof(**input));
    checkMallocError(*input);

    // Data output
    *outputParallel = (data_t*)malloc(tableLen * sizeof(**outputParallel));
    checkMallocError(*outputParallel);
    *inputSequential = (data_t*)malloc(tableLen * sizeof(**inputSequential));
    checkMallocError(*inputSequential);
    *bufferSequential = (data_t*)malloc(tableLen * sizeof(**bufferSequential));
    checkMallocError(*bufferSequential);
    *outputSequential = (data_t*)malloc(tableLen * sizeof(**outputSequential));
    checkMallocError(*outputSequential);
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);

    // Holds samples and splitters in sequential sample sort
    *samples = (data_t*)malloc(NUM_SAMPLES_SEQUENTIAL * OVERSAMPLING_FACTOR * sizeof(**samples));
    checkMallocError(*samples);
    // Holds bucket sizes and bucket offsets in sequential sample sort
    *bucketSizes = (uint_t*)malloc((NUM_SPLITTERS_SEQUENTIAL + 1) * sizeof(**bucketSizes));
    checkMallocError(*bucketSizes);
    // For each element in array holds, to which bucket it belongs
    *elementBuckets = (uint_t*)malloc(tableLen * sizeof(**elementBuckets));
    checkMallocError(*elementBuckets);

    // Offsets of all global buckets
    *globalBucketOffsets = (uint_t*)malloc((NUM_SAMPLES_PARALLEL + 1) * sizeof(**globalBucketOffsets));
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
    data_t *input, data_t *outputParallel, data_t *inputSequential, data_t *bufferSequential,
    data_t *outputSequential, data_t *outputCorrect, data_t *samples, uint_t *bucketSizes, uint_t *elementBuckets,
    uint_t *globalBucketOffsets, double **timers
)
{
    free(input);
    free(outputParallel);
    free(inputSequential);
    free(bufferSequential);
    free(outputSequential);
    free(outputCorrect);

    free(samples);
    free(bucketSizes);
    free(elementBuckets);

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
    data_t **dataTable, data_t **dataBuffer, data_t **samplesLocal, data_t **samplesGlobal,
    uint_t **localBucketSizes, uint_t **localBucketOffsets, uint_t **globalBucketOffsets, uint_t tableLen
)
{
    uint_t elemsPerInitBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    // If table length is not multiple of number of elements processed by one thread block in initial
    // bitonic sort, than array is padded to that length.
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerInitBitonicSort);
    uint_t localSamplesDistance = (elemsPerInitBitonicSort - 1) / NUM_SAMPLES_PARALLEL + 1;
    uint_t localSamplesLen = (tableLenRoundedUp - 1) / localSamplesDistance + 1;
    // (number of all data blocks (tiles)) * (number buckets generated from NUM_SAMPLES_PARALLEL)
    uint_t localBucketsLen = ((tableLenRoundedUp - 1) / elemsPerInitBitonicSort + 1) * (NUM_SAMPLES_PARALLEL + 1);
    cudaError_t error;

    // Arrays for storing data
    error = cudaMalloc(dataTable, tableLenRoundedUp * sizeof(**dataTable));
    checkCudaError(error);
    error = cudaMalloc(dataBuffer, tableLenRoundedUp * sizeof(**dataBuffer));
    checkCudaError(error);

    // Arrays for storing samples
    error = cudaMalloc(samplesLocal, localSamplesLen * sizeof(**samplesLocal));
    checkCudaError(error);
    error = cudaMalloc(samplesGlobal, NUM_SAMPLES_PARALLEL * sizeof(**samplesGlobal));
    checkCudaError(error);

    // Arrays from bucket bookkeeping
    error = cudaMalloc(localBucketSizes, localBucketsLen * sizeof(**localBucketSizes));
    checkCudaError(error);
    error = cudaMalloc(localBucketOffsets, localBucketsLen * sizeof(**localBucketOffsets));
    checkCudaError(error);
    error = cudaMalloc(globalBucketOffsets, (NUM_SAMPLES_PARALLEL + 1) * sizeof(**globalBucketOffsets));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, data_t *samplesLocal, data_t *samplesGlobal, uint_t *localBucketSizes,
    uint_t *localBucketOffsets, uint_t *globalBucketOffsets
)
{
    cudaError_t error;

    // Arrays for storing data
    error = cudaFree(dataTable);
    checkCudaError(error);
    error = cudaFree(dataBuffer);
    checkCudaError(error);

    // Arrays for storing samples
    error = cudaFree(samplesLocal);
    checkCudaError(error);
    error = cudaFree(samplesGlobal);
    checkCudaError(error);

    // Arrays from bucket bookkeeping
    error = cudaFree(localBucketSizes);
    checkCudaError(error);
    error = cudaFree(localBucketOffsets);
    checkCudaError(error);
    error = cudaFree(globalBucketOffsets);
    checkCudaError(error);
}
