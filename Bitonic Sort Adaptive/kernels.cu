#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


/*---------------------------------------------------------
-------------------------- UTILS --------------------------
-----------------------------------------------------------*/

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

/*
From provided interval and index returns element in table. Index can't be higher than interval span.
*/
__device__ el_t getTableElement(el_t *table, interval_t interval, uint_t index) {
    bool useInterval1 = index >= interval.length0;
    uint_t offset = useInterval1 ? interval.offset1 : interval.offset0;

    index -= useInterval1 ? interval.length0 : 0;
    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;

    return table[offset + index];
}

/*
Finds the index q, which is and index, where the exchanges in the bitonic sequence begin. All
elements after index q have to be exchanged. Bitonic sequence boundaries are provided with interval.

Example: 2, 3, 5, 7 | 8, 7, 3, 1 --> index q = 2 ; (5, 7 and 3, 1 have to be exchanged).
*/
__device__ int binarySearch(el_t* table, interval_t interval, uint_t subBlockHalfLen, bool orderAsc) {
    // Depending which interval is longer, different start and end indexes are used
    int_t indexStart = interval.length0 <= interval.length1 ? 0 : subBlockHalfLen - interval.length1;
    int_t indexEnd = interval.length0 <= interval.length1 ? interval.length0 : subBlockHalfLen;

    while (indexStart < indexEnd) {
        int index = indexStart + (indexEnd - indexStart) / 2;
        el_t el0 = getTableElement(table, interval, index);
        el_t el1 = getTableElement(table, interval, index + subBlockHalfLen);

        if ((el0.key < el1.key) ^ orderAsc) {
            indexStart = index + 1;
        } else {
            indexEnd = index;
        }
    }

    return indexStart;
}


/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

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

/*

*/
__global__ void initIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t step,
    uint_t phasesBitonicMerge) {
    extern __shared__ interval_t intervalsTile[];
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t subBlockSize = 1 << step;
    interval_t interval;

    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
        uint_t offset0 = (blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) * subBlockSize;
        uint_t offset1 = (blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) * subBlockSize + subBlockSize / 2;

        interval.offset0 = (blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) % 2 ? offset1 : offset0;
        interval.offset1 = (blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) % 2 ? offset0 : offset1;
        interval.length0 = subBlockSize / 2;
        interval.length1 = subBlockSize / 2;

        intervalsTile[threadIdx.x] = interval;
    }

    for (int stride = 1; subBlockSize > 1 << phasesBitonicMerge; subBlockSize /= 2, stride *= 2) {
        uint_t isThreadActive = threadIdx.x < tableLen / subBlockSize / gridDim.x;

        if (isThreadActive) {
            interval = intervalsTile[threadIdx.x];
        }
        __syncthreads();

        if (isThreadActive) {
            uint_t q = binarySearch(table, interval, subBlockSize / 2, ((blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) / stride) & 1);
            interval_t newInterval;

            // Left sub-block
            newInterval.offset0 = interval.offset0;
            newInterval.length0 = q;
            newInterval.offset1 = interval.offset1 + interval.length1 - subBlockSize / 2 + q;
            newInterval.length1 = subBlockSize / 2 - q;

            intervalsTile[2 * threadIdx.x] = newInterval;

            // Right sub-block
            newInterval.offset0 = interval.offset0 + q;
            newInterval.length0 = interval.length0 - q;
            newInterval.offset1 = interval.offset1;
            newInterval.length1 = q + interval.length1 - subBlockSize / 2;

            intervalsTile[2 * threadIdx.x + 1] = newInterval;
        }
        __syncthreads();
    }

    intervals[2 * index] = intervalsTile[2 * threadIdx.x];
    intervals[2 * index + 1] = intervalsTile[2 * threadIdx.x + 1];

    /*__syncthreads();
    if (threadIdx.x == 0) {
        uint_t stride = blockIdx.x * 2 * blockDim.x;
        for (int i = stride; i < stride + 4; i++) {
            printf("%d %d [%2d, %2d], [%2d, %2d]\n", blockIdx.x, threadIdx.x,
                intervals[i].offset0, intervals[i].length0, intervals[i].offset1, intervals[i].length1
            );
        }
        printf("\n");
    }*/
}

__global__ void generateIntervalsKernel(el_t *table, interval_t *input, interval_t *output, uint_t tableLen,
                                        uint_t phase, uint_t stepStart, uint_t stepEnd) {
    extern __shared__ interval_t intervalsTile[];
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t subBlockSize = 1 << stepStart;
    interval_t interval;

    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
        intervalsTile[threadIdx.x] = input[blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x];
    }

    for (int stride = 1 << (phase - stepStart); subBlockSize > 1 << stepEnd; subBlockSize /= 2, stride *= 2) {
        uint_t isThreadActive = threadIdx.x < tableLen / subBlockSize / gridDim.x;

        if (isThreadActive) {
            interval = intervalsTile[threadIdx.x];
        }
        __syncthreads();

        if (isThreadActive) {
            uint_t q = binarySearch(table, interval, subBlockSize / 2, ((blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x) / stride) & 1);
            interval_t newInterval;

            // Left sub-block
            newInterval.offset0 = interval.offset0;
            newInterval.length0 = q;
            newInterval.offset1 = interval.offset1 + interval.length1 - subBlockSize / 2 + q;
            newInterval.length1 = subBlockSize / 2 - q;

            intervalsTile[2 * threadIdx.x] = newInterval;

            // Right sub-block
            newInterval.offset0 = interval.offset0 + q;
            newInterval.length0 = interval.length0 - q;
            newInterval.offset1 = interval.offset1;
            newInterval.length1 = q + interval.length1 - subBlockSize / 2;

            intervalsTile[2 * threadIdx.x + 1] = newInterval;
        }
        __syncthreads();
    }

    output[2 * index] = intervalsTile[2 * threadIdx.x];
    output[2 * index + 1] = intervalsTile[2 * threadIdx.x + 1];

    /*__syncthreads();
    if (threadIdx.x == 0) {
        uint_t stride = blockIdx.x * 2 * blockDim.x;
        for (int i = stride; i < stride + 4; i++) {
            printf("%d %d [%2d, %2d], [%2d, %2d]\n", blockIdx.x, threadIdx.x,
                output[i].offset0, output[i].length0, output[i].offset1, output[i].length1
            );
        }
        printf("\n");
    }*/
}

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
*/
__global__ void bitonicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t phase, bool orderAsc) {
    extern __shared__ el_t mergeTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    interval_t interval = intervals[blockIdx.x];
    // Elements inside same sub-block have to be ordered in same direction
    bool direction = orderAsc ^ ((index >> phase) & 1);

    // Every thread loads 2 elements
    mergeTile[threadIdx.x] = getTableElement(input, interval, threadIdx.x);
    mergeTile[blockDim.x + threadIdx.x] = getTableElement(input, interval, blockDim.x + threadIdx.x);

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
