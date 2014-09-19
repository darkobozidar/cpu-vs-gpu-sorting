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

__device__ int binarySearch(el_t* table, interval_t *intervals, uint_t subBlockHalfLen) {
    int_t indexStart = 0;
    int_t indexEnd = intervals[0].len;

    while (indexStart < indexEnd) {
        int index = indexStart + (indexEnd - indexStart) / 2;

        if (getTableElement(table, intervals, index) < getTableElement(table, intervals, index + subBlockHalfLen)) {
            indexStart = index + 1;
        } else {
            indexEnd = index;
        }
    }

    return indexStart;
}

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
    interval_t *tile = intervalsTile + 4 * threadIdx.x;
    uint_t subBlockSize = 1 << step;

    if (threadIdx.x < tableLen / subBlockSize) {
        tile[0].offset = threadIdx.x * subBlockSize;
        tile[0].len = subBlockSize / 2;
        tile[1].offset = threadIdx.x * subBlockSize + subBlockSize / 2;
        tile[1].len = subBlockSize / 2;
    }

    for (; step > phasesBitonicMerge + 1; step--, subBlockSize /= 2) {
        __syncthreads();
        interval_t interval0 = tile[0];
        interval_t interval1 = tile[1];

        __syncthreads();
        uint_t activeThreads = tableLen / (1 << step);

        if (threadIdx.x < activeThreads) {
            uint_t q = binarySearch(table, tile, subBlockSize / 2);

            // Left sub-block
            tile[0].offset = interval0.offset;
            tile[0].len = q;
            tile[1].offset = interval1.offset + interval1.len - subBlockSize / 2 + q;
            tile[1].len = subBlockSize / 2 - q;
            // Right sub-block
            tile[2 * activeThreads].offset = interval1.offset;
            tile[2 * activeThreads].len = q + interval1.len - subBlockSize / 2;
            tile[2 * activeThreads + 1].offset = interval0.offset + q;
            tile[2 * activeThreads + 1].len = interval0.len - q;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            if (i && (i % 2 == 0)) {
                printf("\n");
            }
            if (i && (i % 2 != 0)){
                printf(", ");
            }

            printf("[%2d, %2d]", tile[i].offset, tile[i].len);
        }

        printf("\n\n");
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
