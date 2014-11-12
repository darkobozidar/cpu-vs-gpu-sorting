#include <stdio.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__global__ void printElemsKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i].key);
    }
    printf("\n");
}

__global__ void printDataKernel(data_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i]);
    }
    printf("\n");
}

/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, order_t sortOrder) {
    if (((int_t)(elem1->key - elem2->key) > 0) ^ sortOrder) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

__device__ void compareExchange(data_t *elem1, data_t *elem2, order_t sortOrder) {
    if (((int_t)(*elem1 - *elem2) > 0) ^ sortOrder) {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

template <typename T>
__device__ void bitonicSort(T *dataTable, uint_t tableLen, order_t sortOrder) {
    extern __shared__ T bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1) {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT) {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
                if (stride == subBlockSize) {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength) {
                    break;
                }

                compareExchange(&bitonicSortTile[index], &bitonicSortTile[index + offset], sortOrder);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort and collects samples in array for local samples.
*/
template <typename T>
__global__ void bitonicSortCollectSamplesKernel(T *dataTable, data_t *localSamples, uint_t tableLen,
                                                order_t sortOrder) {
    extern __shared__ T bitonicSortTile[];

    bitonicSort<T>(dataTable, tableLen, sortOrder);

    // After sort has been performed, samples are scattered to array of local samples
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    uint_t localSamplesDistance = elemsPerThreadBlock / NUM_SAMPLES;
    uint_t samplesPerThreadBlock = (dataBlockLength - 1) / localSamplesDistance + 1;

    for (uint_t tx = threadIdx.x; tx < samplesPerThreadBlock; tx += THREADS_PER_BITONIC_SORT) {
        localSamples[blockIdx.x * NUM_SAMPLES + tx] = bitonicSortTile[tx * localSamplesDistance].key;
    }
}

template __global__ void bitonicSortCollectSamplesKernel<el_t>(
    el_t *dataTable, data_t *localSamples, uint_t tableLen, order_t sortOrder
);

/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory.
*/
template <typename T>
__global__ void bitonicMergeGlobalKernel(T *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase,
                                         order_t sortOrder) {
    uint_t stride = 1 << (step - 1);
    uint_t pairsPerThreadBlock = (THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE) >> 1;
    uint_t indexGlobal = blockIdx.x * pairsPerThreadBlock + threadIdx.x;

    for (uint_t i = 0; i < ELEMS_PER_THREAD_GLOBAL_MERGE >> 1; i++) {
        uint_t indexThread = indexGlobal + i * THREADS_PER_GLOBAL_MERGE;
        uint_t offset = stride;

        // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
        if (firstStepOfPhase) {
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
        if (index + offset >= tableLen) {
            break;
        }

        compareExchange(&dataTable[index], &dataTable[index + offset], sortOrder);
    }
}

template __global__ void bitonicMergeGlobalKernel<data_t>(
    data_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase, order_t sortOrder
);
template __global__ void bitonicMergeGlobalKernel<el_t>(
    el_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase, order_t sortOrder
);

/*
Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <typename T>
__global__ void bitonicMergeLocalKernel(T *dataTable, uint_t tableLen, uint_t step, bool isFirstStepOfPhase,
                                        order_t sortOrder) {
    __shared__ T mergeTile[THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE];

    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
    uint_t pairsPerBlockLength = dataBlockLength >> 1;

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
        mergeTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1) {
        for (uint_t tx = threadIdx.x; tx < pairsPerBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
            uint_t indexThread = tx;
            uint_t offset = stride;

            // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
            if (isFirstStepOfPhase) {
                offset = ((tx & (stride - 1)) << 1) + 1;
                indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                isFirstStepOfPhase = false;
            }

            uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
            if (index + offset >= dataBlockLength) {
                break;
            }

            compareExchange(&mergeTile[index], &mergeTile[index + offset], sortOrder);
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
        dataTable[offset + tx] = mergeTile[tx];
    }
}

template __global__ void bitonicMergeLocalKernel<data_t>(
    data_t *dataTable, uint_t tableLen, uint_t step, bool isFirstStepOfPhase, order_t sortOrder
);
template __global__ void bitonicMergeLocalKernel<el_t>(
    el_t *dataTable, uint_t tableLen, uint_t step, bool isFirstStepOfPhase, order_t sortOrder
);

/*
From LOCAL samples extracts GLOBAL samples (every NUM_SAMPLES sample). This is done by one thread block.
*/
__global__ void collectGlobalSamplesKernel(data_t *samples, uint_t samplesLen) {
    // Shared memory is needed, because samples are read and written to the same array (race condition).
    __shared__ data_t globalSamplesTile[NUM_SAMPLES];
    uint_t samplesDistance = samplesLen / NUM_SAMPLES;

    // We also add (samplesDistance / 2) to collect samples as evenly as possible
    globalSamplesTile[threadIdx.x] = samples[threadIdx.x * samplesDistance + (samplesDistance / 2)];
    __syncthreads();
    samples[threadIdx.x] = globalSamplesTile[threadIdx.x];
}

__device__ int binarySearchInclusive(el_t* dataTable, data_t target, int_t indexStart, int_t indexEnd,
                                     order_t sortOrder) {
    while (indexStart <= indexEnd) {
        // Floor to multiplier of stride - needed for strides > 1
        int index = (indexStart + indexEnd) / 2;

        if ((target <= dataTable[index].key) ^ (sortOrder)) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

// TODO check if it is better, to read data chunks into shared memory and have one thread block per one data block
__global__ void sampleIndexingKernel(el_t *dataTable, const data_t* __restrict__ samples, data_t* samplesBuffer,
                                     uint_t tableLen, order_t sortOrder) {
    __shared__ uint_t indexingTile[THREADS_PER_SAMPLE_INDEXING];

    uint_t sampleIndex = threadIdx.x % NUM_SAMPLES;
    data_t sample = samples[sampleIndex];

    // One thread block can process multiple data blocks (multiple chunks of data previously sorted by bitonic sort).
    uint_t dataBlocksPerThreadBlock = THREADS_PER_SAMPLE_INDEXING / NUM_SAMPLES;
    uint_t dataBlockIndex = threadIdx.x / NUM_SAMPLES;
    uint_t elemsPerBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;

    uint_t indexBlock = (blockIdx.x * dataBlocksPerThreadBlock + dataBlockIndex);
    uint_t offset = indexBlock * elemsPerBitonicSort;
    uint_t dataBlockLength = offset + elemsPerBitonicSort <= tableLen ? elemsPerBitonicSort : tableLen - offset;

    indexingTile[threadIdx.x] = binarySearchInclusive(
        dataTable, sample, offset, offset + dataBlockIndex, sortOrder
    );
    __syncthreads();

    // TODO check if can be done withouth this extra step
    uint_t prevIndex = 0;
    if (threadIdx.x > 0) {
        prevIndex = indexingTile[threadIdx.x - 1];
    }
    __syncthreads();

    uint_t outputSampleIndex = sampleIndex * (gridDim.x * dataBlocksPerThreadBlock * NUM_SAMPLES) + indexBlock;
    samplesBuffer[outputSampleIndex] = indexingTile[threadIdx.x] - prevIndex;
}
