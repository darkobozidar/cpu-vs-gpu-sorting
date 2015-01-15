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
    data_t **inputSequentialKeys, data_t **inputSequentialValues, data_t **bufferSequentialKeys,
    data_t **bufferSequentialValues, data_t **outputSequentialKeys, data_t **outputSequentialValues,
    data_t **outputCorrect, data_t **samples, uint_t **elementBuckets, uint_t **globalBucketOffsets,
    double ***timers, uint_t tableLen, uint_t testRepetitions
)
{
    // Data input
    *inputKeys = (data_t*)malloc(tableLen * sizeof(**inputKeys));
    checkMallocError(*inputKeys);
    *inputValues = (data_t*)malloc(tableLen * sizeof(**inputValues));
    checkMallocError(*inputValues);

    // Data container for PARALLEL sort
    *outputParallelKeys = (data_t*)malloc(tableLen * sizeof(**outputParallelKeys));
    checkMallocError(*outputParallelKeys);
    *outputParallelValues = (data_t*)malloc(tableLen * sizeof(**outputParallelValues));
    checkMallocError(*outputParallelValues);

    // Data container for SEQUENTIAL sort
    *inputSequentialKeys = (data_t*)malloc(tableLen * sizeof(**inputSequentialKeys));
    checkMallocError(*inputSequentialKeys);
    *inputSequentialValues = (data_t*)malloc(tableLen * sizeof(**inputSequentialValues));
    checkMallocError(*inputSequentialValues);
    *bufferSequentialKeys = (data_t*)malloc(tableLen * sizeof(**bufferSequentialKeys));
    checkMallocError(*bufferSequentialKeys);
    *bufferSequentialValues = (data_t*)malloc(tableLen * sizeof(**bufferSequentialValues));
    checkMallocError(*bufferSequentialValues);
    *outputSequentialKeys = (data_t*)malloc(tableLen * sizeof(**outputSequentialKeys));
    checkMallocError(*outputSequentialKeys);
    *outputSequentialValues = (data_t*)malloc(tableLen * sizeof(**outputSequentialValues));
    checkMallocError(*outputSequentialValues);

    // Data containser for CORRECT sort
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);

    // Holds samples and splitters in sequential sample sort (needed for sequential sample sort)
    *samples = (data_t*)malloc(NUM_SAMPLES_SEQUENTIAL * OVERSAMPLING_FACTOR * sizeof(**samples));
    checkMallocError(*samples);
    // For each element in array holds, to which bucket it belongs (needed for sequential sample sort)
    *elementBuckets = (uint_t*)malloc(tableLen * sizeof(**elementBuckets));
    checkMallocError(*elementBuckets);

    // Offsets of all global buckets (Needed for parallel sort)
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
    data_t *inputKeys, data_t *inputValues, data_t *outputParallelKeys, data_t *outputParallelValues,
    data_t *inputSequentialKeys, data_t *inputSequentialValues, data_t *bufferSequentialKeys,
    data_t *bufferSequentialValues, data_t *outputSequentialKeys, data_t *outputSequentialValues,
    data_t *outputCorrect, data_t *samples, uint_t *elementBuckets, uint_t *globalBucketOffsets, double **timers
)
{
    // Data input
    free(inputKeys);
    free(inputValues);

    // Data container for PARALLEL sort
    free(outputParallelKeys);
    free(outputParallelValues);

    // Data containser for SEQUENTIAL sort
    free(inputSequentialKeys);
    free(inputSequentialValues);
    free(bufferSequentialKeys);
    free(bufferSequentialValues);
    free(outputSequentialKeys);
    free(outputSequentialValues);

    // Data container for CORRECT sort
    free(outputCorrect);

    // Bookkeeping arrays needed for SEQUENTIAL sample sort
    free(samples);
    free(elementBuckets);

    // Bookkeeping arrays needed for PARALLEL sample sort
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
    data_t **dataTableKeys, data_t **dataTableValues, data_t **d_dataBufferKeys, data_t **d_dataBufferValues,
    data_t **samplesLocal, data_t **samplesGlobal, uint_t **localBucketSizes, uint_t **localBucketOffsets,
    uint_t **globalBucketOffsets, uint_t tableLen
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
    error = cudaMalloc(dataTableKeys, tableLenRoundedUp * sizeof(**dataTableKeys));
    checkCudaError(error);
    error = cudaMalloc(dataTableValues, tableLenRoundedUp * sizeof(**dataTableValues));
    checkCudaError(error);
    error = cudaMalloc(d_dataBufferKeys, tableLenRoundedUp * sizeof(**d_dataBufferKeys));
    checkCudaError(error);
    error = cudaMalloc(d_dataBufferValues, tableLenRoundedUp * sizeof(**d_dataBufferValues));
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
    data_t *dataTableKeys, data_t *dataTableValues, data_t *d_dataBufferKeys, data_t *d_dataBufferValues,
    data_t *samplesLocal, data_t *samplesGlobal, uint_t *localBucketSizes, uint_t *localBucketOffsets,
    uint_t *globalBucketOffsets
)
{
    cudaError_t error;

    // Arrays for storing data
    error = cudaFree(dataTableKeys);
    checkCudaError(error);
    error = cudaFree(dataTableValues);
    checkCudaError(error);
    error = cudaFree(d_dataBufferKeys);
    checkCudaError(error);
    error = cudaFree(d_dataBufferValues);
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
