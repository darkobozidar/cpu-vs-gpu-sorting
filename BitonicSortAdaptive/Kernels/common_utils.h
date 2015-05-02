#ifndef KERNELS_COMMON_UTILS_BITONIC_SORT_ADAPTIVE_H
#define KERNELS_COMMON_UTILS_BITONIC_SORT_ADAPTIVE_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../data_types.h"
#include "common_utils.h"


/*
From provided interval and index returns element in array. Index can't be greater than interval span.
*/
__device__ data_t getArrayKey(data_t *table, interval_t interval, uint_t index)
{
    bool useInterval1 = index >= interval.length0;
    uint_t offset = useInterval1 ? interval.offset1 : interval.offset0;

    index -= useInterval1 ? interval.length0 : 0;
    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;

    return table[offset + index];
}

/*
From provided interval and index returns element in array. Index can't be greater than interval span.
*/
__device__ void getArrayKeyValue(
    data_t *keys, data_t *values, interval_t interval, uint_t index, data_t *key, data_t *value
)
{
    bool useInterval1 = index >= interval.length0;
    uint_t offset = useInterval1 ? interval.offset1 : interval.offset0;

    index -= useInterval1 ? interval.length0 : 0;
    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;
    index += offset;

    *key = keys[index];
    *value = values[index];
}

/*
Finds the index q, which is an index, where the exchanges in the bitonic sequence begin. All
elements after index q have to be exchanged. Bitonic sequence boundaries are provided with interval.

Example: 2, 3, 5, 7 | 8, 7, 3, 1 --> index q = 2 ; (5, 7 and 3, 1 have to be exchanged).
*/
template <order_t sortOrder>
inline __device__ int_t binarySearchInterval(data_t* table, interval_t interval, uint_t subBlockHalfLen)
{
    // Depending which interval is longer, different start and end indexes are used
    int_t indexStart = interval.length0 <= interval.length1 ? 0 : subBlockHalfLen - interval.length1;
    int_t indexEnd = interval.length0 <= interval.length1 ? interval.length0 : subBlockHalfLen;

    while (indexStart < indexEnd)
    {
        int index = indexStart + (indexEnd - indexStart) / 2;
        data_t el0 = getArrayKey(table, interval, index);
        data_t el1 = getArrayKey(table, interval, index + subBlockHalfLen);

        if ((sortOrder == ORDER_ASC) ? (el0 > el1) : (el0 < el1))
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
Generates intervals in provided array until size of sub block is grater than end sub block size.
Sub block size is the size of one block in bitonic merge step.
*/
template <order_t sortOrder, uint_t elementsPerThread>
inline __device__ void generateIntervals(
    data_t *table, uint_t subBlockHalfSize, uint_t subBlockSizeEnd, uint_t stride, uint_t activeThreadsPerBlock
)
{
    extern __shared__ interval_t intervalsTile[];
    interval_t interval;

    // Only active threads have to generate intervals. This increases by 2 in every iteration. If threads were
    // reading and writing in same array, then one additional syncthreads() would be needed in order for all
    // active threads to read their corresponding intervals before generating new. For this purpose buffer is used
    // as output array.
    interval_t *intervals = intervalsTile;
    interval_t *intervalsBuffer = intervalsTile + blockDim.x * elementsPerThread;

    for (; subBlockHalfSize >= subBlockSizeEnd; subBlockHalfSize /= 2, stride *= 2, activeThreadsPerBlock *= 2)
    {
        for (uint_t tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
        {
            interval = intervals[tx];

            uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
            bool orderAsc = sortOrder ^ ((intervalIndex / stride) & 1);
            uint_t q;

            // Finds q - an index, where exchanges begin in bitonic sequences being merged.
            if (orderAsc)
            {
                q = binarySearchInterval<ORDER_ASC>(table, interval, subBlockHalfSize);
            }
            else
            {
                q = binarySearchInterval<ORDER_DESC>(table, interval, subBlockHalfSize);
            }

            // Output indexes of newly generated intervals
            uint_t index1 = 2 * tx;
            uint_t index2 = index1 + 1;

            // Left sub-block
            intervalsBuffer[index1].offset0 = interval.offset0;
            intervalsBuffer[index1].length0 = q;
            intervalsBuffer[index1].offset1 = interval.offset1 + interval.length1 - subBlockHalfSize + q;
            intervalsBuffer[index1].length1 = subBlockHalfSize - q;

            // Right sub-block. Intervals are reversed, otherwise intervals aren't generated correctly.
            intervalsBuffer[index2].offset0 = interval.offset0 + q;
            intervalsBuffer[index2].length0 = interval.length0 - q;
            intervalsBuffer[index2].offset1 = interval.offset1;
            intervalsBuffer[index2].length1 = q + interval.length1 - subBlockHalfSize;
        }

        interval_t *temp = intervals;
        intervals = intervalsBuffer;
        intervalsBuffer = temp;

        __syncthreads();
    }
}

#endif
