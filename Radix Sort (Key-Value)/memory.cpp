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
    data_t **inputSequentialKeys, data_t **inputSequentialValues, data_t **outputSequentialKeys,
    data_t **outputSequentialValues, data_t **outputCorrect, uint_t **countersSequential, double ***timers,
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
    *inputSequentialKeys = (data_t*)malloc(tableLen * sizeof(**inputSequentialKeys));
    checkMallocError(*inputSequentialKeys);
    *inputSequentialValues = (data_t*)malloc(tableLen * sizeof(**inputSequentialValues));
    checkMallocError(*inputSequentialValues);
    *outputSequentialKeys = (data_t*)malloc(tableLen * sizeof(**outputSequentialKeys));
    checkMallocError(*outputSequentialKeys);
    *outputSequentialValues = (data_t*)malloc(tableLen * sizeof(**outputSequentialValues));
    checkMallocError(*outputSequentialValues);
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);

    // Counters of element occurances - needed for sequential radix sort
    *countersSequential = (data_t*)malloc(RADIX_SEQUENTIAL * sizeof(**countersSequential));
    checkMallocError(*countersSequential);

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
    data_t *inputSequentialKeys, data_t *inputSequentialValues, data_t *outputSequentialKeys,
    data_t *outputSequentialValues, data_t *outputCorrect, uint_t *countersSequential, double **timers
)
{
    free(inputKeys);
    free(inputValues);
    free(outputParallelKeys);
    free(outputParallelValues);
    free(inputSequentialKeys);
    free(inputSequentialValues);
    free(outputSequentialKeys);
    free(outputSequentialValues);
    free(outputCorrect);
    free(countersSequential);

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
    uint_t **bucketOffsetsLocal, uint_t **bucketOffsetsGlobal, uint_t **bucketSizes, uint_t tableLen
)
{
    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    uint_t bucketsLen = RADIX_PARALLEL * ((tableLen - 1) / elemsPerLocalSort + 1);
    // In case table length not divisable by number of elements processed by one thread block in local radix
    // sort, data table is padded to the next multiple of number of elements per local radix sort.
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerLocalSort);
    cudaError_t error;

    error = cudaMalloc(dataTableKeys, tableLenRoundedUp * sizeof(**dataTableKeys));
    checkCudaError(error);
    error = cudaMalloc(dataTableValues, tableLenRoundedUp * sizeof(**dataTableValues));
    checkCudaError(error);
    error = cudaMalloc(dataBufferKeys, tableLenRoundedUp * sizeof(**dataBufferKeys));
    checkCudaError(error);
    error = cudaMalloc(dataBufferValues, tableLenRoundedUp * sizeof(**dataBufferValues));
    checkCudaError(error);

    error = cudaMalloc(bucketOffsetsLocal, bucketsLen * sizeof(**bucketOffsetsLocal));
    checkCudaError(error);
    error = cudaMalloc(bucketOffsetsGlobal, bucketsLen * sizeof(**bucketOffsetsGlobal));
    checkCudaError(error);
    error = cudaMalloc(bucketSizes, bucketsLen * sizeof(**bucketSizes));
    checkCudaError(error);
}

/*
Frees device memory.
*/
void freeDeviceMemory(
    data_t *dataTableKeys, data_t *dataTableValues, data_t *dataBufferKeys, data_t *dataBufferValues,
    uint_t *bucketOffsetsLocal, uint_t *bucketOffsetsGlobal, uint_t *bucketSizes
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

    error = cudaFree(bucketOffsetsLocal);
    checkCudaError(error);
    error = cudaFree(bucketOffsetsGlobal);
    checkCudaError(error);
    error = cudaFree(bucketSizes);
    checkCudaError(error);
}
