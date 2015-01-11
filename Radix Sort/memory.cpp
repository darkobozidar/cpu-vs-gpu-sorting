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
    data_t **input, data_t **outputParallel, data_t **h_inputSequential, data_t **outputSequential,
    data_t **outputCorrect, uint_t **countersSequential, double ***timers, uint_t tableLen, uint_t testRepetitions
)
{
    // Data input
    *input = (data_t*)malloc(tableLen * sizeof(**input));
    checkMallocError(*input);

    // Data output
    *outputParallel = (data_t*)malloc(tableLen * sizeof(**outputParallel));
    checkMallocError(*outputParallel);
    *h_inputSequential = (data_t*)malloc(tableLen * sizeof(**h_inputSequential));
    checkMallocError(*h_inputSequential);
    *outputSequential = (data_t*)malloc(tableLen * sizeof(**outputSequential));
    checkMallocError(*outputSequential);
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
    data_t *input, data_t *outputParallel, data_t *h_inputSequential, data_t *outputSequential, data_t *outputCorrect,
    uint_t *countersSequential, double **timers
)
{
    free(input);
    free(outputParallel);
    free(h_inputSequential);
    free(outputSequential);
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
    data_t **dataTable, data_t **dataBuffer, uint_t **bucketOffsetsLocal, uint_t **bucketOffsetsGlobal,
    uint_t **bucketSizes, uint_t tableLen
)
{
    uint_t threadsPerSortLocal = min((tableLen - 1) / ELEMS_PER_THREAD_LOCAL + 1, THREADS_PER_LOCAL_SORT);
    uint_t bucketsLen = RADIX_PARALLEL * ((tableLen - 1) / (ELEMS_PER_THREAD_LOCAL * threadsPerSortLocal) + 1);
    // In case table length is not power of 2, data table is padded to the next multiple of number of elements
    // per local radix sort.
    uint_t tableLenRoundedUp = roundUp(tableLen, threadsPerSortLocal);
    cudaError_t error;

    error = cudaMalloc(dataTable, tableLenRoundedUp * sizeof(**dataTable));
    checkCudaError(error);
    error = cudaMalloc(dataBuffer, tableLenRoundedUp * sizeof(**dataBuffer));
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
    data_t *dataTable, data_t *dataBuffer, uint_t *bucketOffsetsLocal, uint_t *bucketOffsetsGlobal,
    uint_t *bucketSizes
)
{
    cudaError_t error;

    error = cudaFree(dataTable);
    checkCudaError(error);
    error = cudaFree(dataBuffer);
    checkCudaError(error);

    error = cudaFree(bucketOffsetsLocal);
    checkCudaError(error);
    error = cudaFree(bucketOffsetsGlobal);
    checkCudaError(error);
    error = cudaFree(bucketSizes);
    checkCudaError(error);
}
