#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, bool orderAsc) {
    if (((int_t)(elem1->key - elem2->key) <= 0) ^ orderAsc) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

__device__ uint_t getTableElement(el_t *table, interval_t *intervals, uint_t index) {
    uint_t i = 0;
    while (index >= intervals[i].len) {
        index -= intervals[i].len;
        i++;
    }

    return table[intervals[i].offset + index].key;
}

//__device__ int binarySearchExclusive(el_t* table, el_t target, int_t indexStart, int_t indexEnd,
//    uint_t stride, bool orderAsc) {
//    while (indexStart <= indexEnd) {
//        // Floor to multiplier of stride - needed for strides > 1
//        int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ ULONG_MAX);
//
//        if ((target.key < table[index].key) ^ (!orderAsc)) {
//            indexEnd = index - stride;
//        }
//        else {
//            indexStart = index + stride;
//        }
//    }
//
//    return indexStart;
//}
//
//__device__ int binarySearchInclusive(el_t* table, el_t target, int_t indexStart, int_t indexEnd,
//    uint_t stride, bool orderAsc) {
//    while (indexStart <= indexEnd) {
//        // Floor to multiplier of stride - needed for strides > 1
//        int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ ULONG_MAX);
//
//        if ((target.key <= table[index].key) ^ (!orderAsc)) {
//            indexEnd = index - stride;
//        }
//        else {
//            indexStart = index + stride;
//        }
//    }
//
//    return indexStart;
//}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
__global__ void bitonicSortKernel(el_t *table, bool orderAsc) {
    extern __shared__ el_t sortTile[];
    // If shared memory size is lower than table length, than every block has to be ordered
    // in opposite direction -> bitonic sequence.
    bool blockDirection = orderAsc ^ (blockIdx.x & 1);

    // Every thread loads 2 elements
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sortTile[threadIdx.x] = table[index];
    sortTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize <= blockDim.x; subBlockSize <<= 1) {
        bool direction = blockDirection ^ ((threadIdx.x & subBlockSize) != 0);

        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            __syncthreads();
            uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareExchange(&sortTile[start], &sortTile[start + stride], direction);
        }
    }

    __syncthreads();
    table[index] = sortTile[threadIdx.x];
    table[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}

__global__ void generateIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t step,
                                        uint_t phasesBitonicMerge) {
    extern __shared__ interval_t intervalsTile[];
    interval_t *tile = intervalsTile + 2 * threadIdx.x;
    uint_t subBlockSize = 1 << step;
    interval_t interval0;
    interval_t interval1;

    if (threadIdx.x + 1 <= tableLen / subBlockSize) {
        interval0.offset = threadIdx.x * subBlockSize;
        interval0.len = subBlockSize / 2;
        interval1.offset = threadIdx.x * subBlockSize + subBlockSize / 2;
        interval1.len = subBlockSize / 2;

        tile[0] = interval0;
        tile[1] = interval1;
    }

    for (; step > phasesBitonicMerge; step--) {
        __syncthreads();

        if (threadIdx.x + 1 <= tableLen / (1 << step)) {
            interval0 = tile[threadIdx.x];
            interval1 = tile[threadIdx.x + blockDim.x];
        }
    }
}

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
*/
__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc) {
    extern __shared__ el_t mergeTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    // Elements inside same sub-block have to be ordered in same direction
    bool direction = orderAsc ^ ((index >> phase) & 1);

    // Every thread loads 2 elements
    mergeTile[threadIdx.x] = table[index];
    mergeTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];

    // Bitonic merge
    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&mergeTile[start], &mergeTile[start + stride], direction);
    }

    __syncthreads();
    table[index] = mergeTile[threadIdx.x];
    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
