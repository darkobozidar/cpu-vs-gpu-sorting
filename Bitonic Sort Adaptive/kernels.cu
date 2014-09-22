#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


__global__ void printTableKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i]);
    }
    printf("\n\n");
}

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
From provided interval and index returns element in table.
*/
__device__ el_t getTableElement(el_t *table, interval_t interval, uint_t index) {
    bool useInterval1 = index >= interval.length0;
    uint_t offset = useInterval1 ? interval.offset1 : interval.offset0;

    index -= useInterval1 ? interval.length0 : 0;
    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;

    return table[offset + index];
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

// TODO binary search in opposite side
__device__ int binarySearch(el_t* table, interval_t interval, uint_t subBlockHalfLen, bool bla) {
    int_t indexStart = 0;
    int_t indexEnd = interval.length0 < subBlockHalfLen ? interval.length0 : subBlockHalfLen;

    while (indexStart < indexEnd) {
        int index = indexStart + (indexEnd - indexStart) / 2;
        el_t el0 = getTableElement(table, interval, index);
        el_t el1 = getTableElement(table, interval, index + subBlockHalfLen);

        if (!bla && (el0.key <= el1.key) || bla && (el0.key >= el1.key)) {
            indexStart = index + 1;
        }
        else {
            indexEnd = index;
        }
    }

    return indexStart;
}

__global__ void initIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t step,
    uint_t phasesBitonicMerge) {
    extern __shared__ interval_t intervalsTile[];
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t subBlockSize = 1 << step;
    interval_t interval;

    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
        uint_t offset0 = index / gridDim.x * subBlockSize;
        uint_t offset1 = index / gridDim.x * subBlockSize + subBlockSize / 2;

        interval.offset0 = index % 2 ? offset1 : offset0;
        interval.offset1 = index % 2 ? offset0 : offset1;
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
            uint_t q = binarySearch(table, interval, subBlockSize / 2, (index / stride) & 1);
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
        for (int i = blockDim.x * blockIdx.x * 2; i < 4; i++) {
            printf("[%2d, %2d], [%2d, %2d]\n",
                intervals[i].offset0, intervals[i].length0, intervals[i].offset1, intervals[i].length1
            );
        }

        printf("\n\n");
    }*/
}

__global__ void generateIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t phase,
                                        uint_t step, uint_t phasesBitonicMerge) {
    extern __shared__ interval_t intervalsTile[];
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t subBlockSize = 1 << step;
    interval_t interval;

    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
        intervalsTile[threadIdx.x] = intervals[blockIdx.x * (tableLen / subBlockSize / gridDim.x) + threadIdx.x];
    }

    for (int stride = 1 << (phase - step); subBlockSize > 1 << phasesBitonicMerge; subBlockSize /= 2, stride *= 2) {
        uint_t isThreadActive = threadIdx.x < tableLen / subBlockSize / gridDim.x;

        if (isThreadActive) {
            interval = intervalsTile[threadIdx.x];
        }
        __syncthreads();

        if (isThreadActive) {
            uint_t q = binarySearch(table, interval, subBlockSize / 2, (index / stride) & 1);
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
        for (int i = blockDim.x * blockIdx.x * 2; i < 8; i++) {
            printf("[%2d, %2d], [%2d, %2d]\n",
                intervals[i].offset0, intervals[i].length0,
                intervals[i].offset1, intervals[i].length1
            );
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
