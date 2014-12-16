#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"
#include "data_types.h"


/*---------------------------------------------------------
-------------------------- UTILS --------------------------
-----------------------------------------------------------*/

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
__device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
From provided interval and index returns element in table. Index can't be higher than interval span.
*/
__device__ data_t getTableElement(data_t *table, interval_t interval, uint_t index)
{
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
template <order_t sortOrder>
__device__ int_t binarySearch(data_t* table, interval_t interval, uint_t subBlockHalfLen)
{
    // Depending which interval is longer, different start and end indexes are used
    int_t indexStart = interval.length0 <= interval.length1 ? 0 : subBlockHalfLen - interval.length1;
    int_t indexEnd = interval.length0 <= interval.length1 ? interval.length0 : subBlockHalfLen;

    while (indexStart < indexEnd)
    {
        int index = indexStart + (indexEnd - indexStart) / 2;
        data_t el0 = getTableElement(table, interval, index);
        data_t el1 = getTableElement(table, interval, index + subBlockHalfLen);

        if ((el0 > el1) ^ sortOrder)
        {
            indexStart = index + 1;
        }
        else
        {
            indexEnd = index;
        }
    }

    return indexStart;
}

/*
Generates intervals in provided table until size of sub block is grater than end sub block size.
Sub block size is the size of one block in bitonic merge step.
*/
template <order_t sortOrder>
__device__ void generateIntervals(
    data_t *table, uint_t subBlockSize, uint_t subBlockSizeEnd, uint_t stride, uint_t activeThreadsPerBlock,
    uint_t elemsPerThread
)
{
    extern __shared__ interval_t intervalsTile[];
    interval_t *intervals = intervalsTile;
    interval_t *intervalsBuffer = intervalsTile + blockDim.x * elemsPerThread;
    interval_t interval;

    for (; subBlockSize > subBlockSizeEnd; subBlockSize /= 2, stride *= 2, activeThreadsPerBlock *= 2)
    {
        for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
        {
            interval = intervals[tx];

            uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
            bool orderAsc = sortOrder ^ ((intervalIndex / stride) & 1);
            uint_t q;

            if (orderAsc)
            {
                q = binarySearch<ORDER_ASC>(table, interval, subBlockSize / 2);
            }
            else
            {
                q = binarySearch<ORDER_DESC>(table, interval, subBlockSize / 2);
            }

            uint_t index1 = 2 * tx;
            uint_t index2 = index1 + 1;

            // Left sub-block
            intervalsBuffer[index1].offset0 = interval.offset0;
            intervalsBuffer[index1].length0 = q;
            intervalsBuffer[index1].offset1 = interval.offset1 + interval.length1 - subBlockSize / 2 + q;
            intervalsBuffer[index1].length1 = subBlockSize / 2 - q;

            // Right sub-block. Intervals are reversed.
            intervalsBuffer[index2].offset0 = interval.offset0 + q;
            intervalsBuffer[index2].length0 = interval.length0 - q;
            intervalsBuffer[index2].offset1 = interval.offset1;
            intervalsBuffer[index2].length1 = q + interval.length1 - subBlockSize / 2;
        }

        interval_t *temp = intervals;
        intervals = intervalsBuffer;
        intervalsBuffer = temp;

        __syncthreads();
    }
}


/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

/*
Adds the padding to table from start index (original table length, which is not power of 2) to the end of the
extended array (which is the next power of 2 of the original table length). Needed because of bitonic sort, for
which table length divisable by 2 is needed.
*/
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        uint_t index = offset + tx;
        dataTable[index] = value;
        dataBuffer[index] = value;
    }
}

template __global__ void addPaddingKernel<MIN_VAL>(
    data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length
);
template __global__ void addPaddingKernel<MAX_VAL>(
    data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length
);


/*
Sorts sub-blocks of input data with bitonic sort.
*/
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t sortTile[];
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;

    // If shared memory size is lower than table length, than adjecent blocks have to be ordered in opposite
    // direction in order to create bitonic sequences.
    bool blockDirection = sortOrder ^ (blockIdx.x & 1);

    // Loads data into shared memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_BITONIC_SORT)
    {
        sortTile[tx] = dataTable[offset + tx];
    }

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize < elemsPerThreadBlock; subBlockSize <<= 1)
    {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            __syncthreads();
            for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += THREADS_PER_BITONIC_SORT)
            {
                bool direction = blockDirection ^ ((tx & subBlockSize) != 0);
                uint_t start = 2 * tx - (tx & (stride - 1));

                if (direction)
                {
                    compareExchange<ORDER_DESC>(&sortTile[start], &sortTile[start + stride]);
                }
                else
                {
                    compareExchange<ORDER_ASC>(&sortTile[start], &sortTile[start + stride]);
                }
            }
        }
    }

    // Stores sorted elements from shared to global memory
    __syncthreads();
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = sortTile[tx];
    }
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen);


/*
Generates initial intervals and continues to evolve them until the end step.
*/
template <order_t sortOrder>
__global__ void initIntervalsKernel(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
)
{
    extern __shared__ interval_t intervalsTile[];
    uint_t subBlockSize = 1 << stepStart;
    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
    uint_t elemsPerThreadBlock = blockDim.x * ELEMS_PER_INIT_INTERVALS;

    for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
        uint_t offset0 = intervalIndex * subBlockSize;
        uint_t offset1 = intervalIndex * subBlockSize + subBlockSize / 2;

        // In every odd block intervals have to be rotated
        intervalsTile[tx].offset0 = intervalIndex % 2 ? offset1 : offset0;
        intervalsTile[tx].offset1 = intervalIndex % 2 ? offset0 : offset1;
        intervalsTile[tx].length0 = subBlockSize / 2;
        intervalsTile[tx].length1 = subBlockSize / 2;
    }
    __syncthreads();

    generateIntervals<sortOrder>(
        table, subBlockSize, 1 << stepEnd, 1, activeThreadsPerBlock, ELEMS_PER_INIT_INTERVALS
    );

    // Calculates offset in global intervals array
    interval_t *outputIntervalsGlobal = intervals + blockIdx.x * elemsPerThreadBlock;
    // Depending if the number of repetitions is divisable by 2, generated intervals are located in FIRST half
    // OR in SECOND half of shared memory (shared memory has 2x size of generated intervals for buffer purposes)
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerThreadBlock : 0);

    // Stores generated intervals from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

template __global__ void initIntervalsKernel<ORDER_ASC>(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
);
template __global__ void initIntervalsKernel<ORDER_DESC>(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
);


/*
Reads the existing intervals from global memory and evolve them until the end step.
*/
template <order_t sortOrder>
__global__ void generateIntervalsKernel(
    data_t *table, interval_t *inputIntervals, interval_t *outputIntervals, uint_t tableLen, uint_t phase,
    uint_t stepStart, uint_t stepEnd
)
{
    extern __shared__ interval_t intervalsTile[];
    uint_t subBlockSize = 1 << stepStart;
    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
    interval_t *inputIntervalsGlobal = inputIntervals + blockIdx.x * activeThreadsPerBlock;

    // Active threads read existing intervals from global memory
    for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        intervalsTile[tx] = inputIntervalsGlobal[tx];
    }
    __syncthreads();

    generateIntervals<sortOrder>(
        table, subBlockSize, 1 << stepEnd, 1 << (phase - stepStart), activeThreadsPerBlock,
        ELEMS_PER_GEN_INTERVALS
    );

    uint_t elemsPerThreadBlock = blockDim.x * ELEMS_PER_GEN_INTERVALS;
    // Calculates offset in global intervals array
    interval_t *outputIntervalsGlobal = outputIntervals + blockIdx.x * elemsPerThreadBlock;
    // Depending if the number of repetitions is divisable by 2, generated intervals are located in FIRST half
    // OR in SECOND half of shared memory (shared memory has 2x size of all generated intervals for buffer purposes)
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerThreadBlock : 0);

    // Stores generated intervals from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

template __global__ void generateIntervalsKernel<ORDER_ASC>(
    data_t *table, interval_t *input, interval_t *output, uint_t tableLen, uint_t phase, uint_t stepStart,
    uint_t stepEnd
);
template __global__ void generateIntervalsKernel<ORDER_DESC>(
    data_t *table, interval_t *input, interval_t *output, uint_t tableLen, uint_t phase, uint_t stepStart,
    uint_t stepEnd
);


/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
*/
template <order_t sortOrder>
__global__ void bitonicMergeKernel(data_t *input, data_t *output, interval_t *intervals, uint_t phase)
{
    extern __shared__ data_t mergeTile[];
    interval_t interval = intervals[blockIdx.x];

    // Elements inside same sub-block have to be ordered in same direction
    uint_t elemsPerThreadBlock = THREADS_PER_MERGE * ELEMS_PER_MERGE;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    bool orderAsc = !sortOrder ^ ((offset >> phase) & 1);

    // Loads data from global to shared memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_MERGE)
    {
        mergeTile[tx] = getTableElement(input, interval, tx);
    }

    // Bitonic merge
    for (uint_t stride = elemsPerThreadBlock / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock / 2; tx += THREADS_PER_MERGE)
        {
            uint_t start = 2 * tx - (tx & (stride - 1));

            if (orderAsc)
            {
                compareExchange<ORDER_ASC>(&mergeTile[start], &mergeTile[start + stride]);
            }
            else
            {
                compareExchange<ORDER_DESC>(&mergeTile[start], &mergeTile[start + stride]);
            }
        }
    }

    // Stores sorted data to buffer array
    __syncthreads();
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_MERGE)
    {
        output[offset + tx] = mergeTile[tx];
    }
}

template __global__ void bitonicMergeKernel<ORDER_ASC>(
    data_t *input, data_t *output, interval_t *intervals, uint_t phase
);
template __global__ void bitonicMergeKernel<ORDER_DESC>(
    data_t *input, data_t *output, interval_t *intervals, uint_t phase
);
