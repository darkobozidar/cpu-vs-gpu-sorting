#include <stdio.h>
#include <math.h>

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

__device__ el_t getTableElement(el_t *table, interval_t *intervals, uint_t index) {
    uint_t i = 0;
    while (index >= intervals[i].len) {
        index -= intervals[i].len;
        i++;
    }

    return table[intervals[i].offset + index];
}

__device__ int binarySearch(el_t* table, interval_t *intervals, uint_t subBlockHalfLen) {
    int_t indexStart = 0;
    int_t indexEnd = intervals[0].len;

    while (indexStart < indexEnd) {
        int index = indexStart + (indexEnd - indexStart) / 2;
        el_t el0 = getTableElement(table, intervals, index);
        el_t el1 = getTableElement(table, intervals, index + subBlockHalfLen);

        // TODO double-check for stability
        if (el0.key < el1.key) {
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
    uint_t index = 2 * threadIdx.x;
    uint_t subBlockSize = 1 << step;

    if (threadIdx.x < tableLen / subBlockSize) {
        intervalsTile[index].offset = threadIdx.x * subBlockSize;
        intervalsTile[index].len = subBlockSize / 2;
        intervalsTile[index + 1].offset = threadIdx.x * subBlockSize + subBlockSize / 2;
        intervalsTile[index + 1].len = subBlockSize / 2;
    }

    for (; step > phasesBitonicMerge; step--, subBlockSize /= 2) {
        // TODO try to put in if statement if possible
        __syncthreads();
        interval_t interval0 = intervalsTile[index];
        interval_t interval1 = intervalsTile[index + 1];

        if (interval0.offset > interval1.offset) {
            interval_t temp = interval0;
            interval0 = interval1;
            interval1 = temp;

            intervalsTile[index] = interval0;
            intervalsTile[index + 1] = interval1;
        }

        __syncthreads();
        uint_t activeThreads = tableLen / (1 << step);

        if (threadIdx.x < activeThreads) {
            uint_t q = binarySearch(table, intervalsTile + index, subBlockSize / 2);

            // Left sub-block
            intervalsTile[2 * index].offset = interval0.offset;
            intervalsTile[2 * index].len = q;
            intervalsTile[2 * index + 1].offset = interval1.offset + interval1.len - subBlockSize / 2 + q;
            intervalsTile[2 * index + 1].len = subBlockSize / 2 - q;
            // Right sub-block
            intervalsTile[2 * index + 2].offset = interval1.offset;
            intervalsTile[2 * index + 2].len = q + interval1.len - subBlockSize / 2;
            intervalsTile[2 * index + 3].offset = interval0.offset + q;
            intervalsTile[2 * index + 3].len = interval0.len - q;
        }
    }

    __syncthreads();
    intervals[index] = intervalsTile[index];
    intervals[index + 1] = intervalsTile[index + 1];

    /*if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            if (i && (i % 2 == 0)) {
                printf("\n");
            }
            if (i && (i % 2 != 0)){
                printf(", ");
            }

            printf("[%2d, %2d]", intervalsTile[i].offset, intervalsTile[i].len);
        }

        printf("\n\n");
    }*/
}

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
*/
__global__ void bitonicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t phase, bool orderAsc) {
    extern __shared__ el_t mergeTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    // Elements inside same sub-block have to be ordered in same direction
    bool direction = orderAsc ^ ((index >> phase) & 1);

    // Every thread loads 2 elements
    mergeTile[threadIdx.x] = getTableElement(input, intervals, index);
    mergeTile[blockDim.x + threadIdx.x] = getTableElement(input, intervals, blockDim.x + index);
    //printf("%2d %2d %2d\n", blockIdx.x, mergeTile[threadIdx.x].key, mergeTile[blockDim.x + threadIdx.x].key);

    // Bitonic merge
    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&mergeTile[start], &mergeTile[start + stride], direction);
    }

    __syncthreads();
    output[index] = mergeTile[threadIdx.x];
    output[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
