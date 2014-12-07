#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"


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

///*
//From provided interval and index returns element in table. Index can't be higher than interval span.
//*/
//__device__ el_t getTableElement(el_t *table, interval_t interval, uint_t index) {
//    bool useInterval1 = index >= interval.length0;
//    uint_t offset = useInterval1 ? interval.offset1 : interval.offset0;
//
//    index -= useInterval1 ? interval.length0 : 0;
//    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;
//
//    return table[offset + index];
//}
//
///*
//Finds the index q, which is and index, where the exchanges in the bitonic sequence begin. All
//elements after index q have to be exchanged. Bitonic sequence boundaries are provided with interval.
//
//Example: 2, 3, 5, 7 | 8, 7, 3, 1 --> index q = 2 ; (5, 7 and 3, 1 have to be exchanged).
//*/
//__device__ int binarySearch(el_t* table, interval_t interval, uint_t subBlockHalfLen, bool orderAsc) {
//    // Depending which interval is longer, different start and end indexes are used
//    int_t indexStart = interval.length0 <= interval.length1 ? 0 : subBlockHalfLen - interval.length1;
//    int_t indexEnd = interval.length0 <= interval.length1 ? interval.length0 : subBlockHalfLen;
//
//    while (indexStart < indexEnd) {
//        int index = indexStart + (indexEnd - indexStart) / 2;
//        el_t el0 = getTableElement(table, interval, index);
//        el_t el1 = getTableElement(table, interval, index + subBlockHalfLen);
//
//        if ((el0.key < el1.key) ^ orderAsc) {
//            indexStart = index + 1;
//        } else {
//            indexEnd = index;
//        }
//    }
//
//    return indexStart;
//}
//
///*
//Generates intervals in provided table until size of sub block is grater than end sub block size.
//Sub block size is the size of one block in bitonic merge step.
//*/
//__device__ void generateIntervals(el_t *table, interval_t *intervals, uint_t subBlockSize, uint_t subBlockSizeEnd,
//                                  uint_t stride, uint_t activeThreadsPerBlock) {
//    interval_t interval;
//
//    for (; subBlockSize > subBlockSizeEnd; subBlockSize /= 2, stride *= 2, activeThreadsPerBlock *= 2) {
//        uint_t isThreadActive = threadIdx.x < activeThreadsPerBlock;
//
//        if (isThreadActive) {
//            interval = intervals[threadIdx.x];
//        }
//        __syncthreads();
//
//        if (isThreadActive) {
//            uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + threadIdx.x;
//            uint_t q = binarySearch(table, interval, subBlockSize / 2, (intervalIndex / stride) & 1);
//
//            // Left sub-block
//            intervals[2 * threadIdx.x].offset0 = interval.offset0;
//            intervals[2 * threadIdx.x].length0 = q;
//            intervals[2 * threadIdx.x].offset1 = interval.offset1 + interval.length1 - subBlockSize / 2 + q;
//            intervals[2 * threadIdx.x].length1 = subBlockSize / 2 - q;
//
//            // Right sub-block
//            intervals[2 * threadIdx.x + 1].offset0 = interval.offset0 + q;
//            intervals[2 * threadIdx.x + 1].length0 = interval.length0 - q;
//            intervals[2 * threadIdx.x + 1].offset1 = interval.offset1;
//            intervals[2 * threadIdx.x + 1].length1 = q + interval.length1 - subBlockSize / 2;
//        }
//        __syncthreads();
//    }
//}


/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT)
            {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In NORMALIZED bitonic sort, first STEP of every PHASE uses different offset than all other
                // STEPS. Also, in first step of every phase, offset sizes are generated in ASCENDING order
                // (normalized bitnic sort requires DESCENDING order). Because of that, we can break the loop if
                // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
                // than thread indexes inside every sub-block have to be reversed.
                if (stride == subBlockSize)
                {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength)
                {
                    break;
                }

                compareExchange<sortOrder>(&bitonicSortTile[index], &bitonicSortTile[index + offset]);
            }

            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen);


///*
//Generates initial intervals and continues to evolve them until the end step.
//*/
//__global__ void initIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart,
//                                    uint_t stepEnd) {
//    extern __shared__ interval_t intervalsTile[];
//    uint_t subBlockSize = 1 << stepStart;
//    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
//    uint_t index;
//
//    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
//        uint_t intervalIndex = blockIdx.x * activeThreadsPerBlock + threadIdx.x;
//        uint_t offset0 = intervalIndex * subBlockSize;
//        uint_t offset1 = intervalIndex * subBlockSize + subBlockSize / 2;
//
//        // In every odd block intervals have to be rotated
//        intervalsTile[threadIdx.x].offset0 = intervalIndex % 2 ? offset1 : offset0;
//        intervalsTile[threadIdx.x].offset1 = intervalIndex % 2 ? offset0 : offset1;
//        intervalsTile[threadIdx.x].length0 = subBlockSize / 2;
//        intervalsTile[threadIdx.x].length1 = subBlockSize / 2;
//    }
//
//    generateIntervals(table, intervalsTile, subBlockSize, 1 << stepEnd, 1, activeThreadsPerBlock);
//
//    index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
//    intervals[index] = intervalsTile[threadIdx.x];
//    intervals[blockDim.x + index] = intervalsTile[blockDim.x + threadIdx.x];
//}
//
///*
//Reads the existing intervals from global memory and evolve them until the end step.
//*/
//__global__ void generateIntervalsKernel(el_t *table, interval_t *input, interval_t *output, uint_t tableLen,
//                                        uint_t phase, uint_t stepStart, uint_t stepEnd) {
//    extern __shared__ interval_t intervalsTile[];
//    uint_t subBlockSize = 1 << stepStart;
//    uint_t activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
//    uint_t index;
//
//    if (threadIdx.x < tableLen / subBlockSize / gridDim.x) {
//        intervalsTile[threadIdx.x] = input[blockIdx.x * activeThreadsPerBlock + threadIdx.x];
//    }
//
//    generateIntervals(
//        table, intervalsTile, subBlockSize, 1 << stepEnd, 1 << (phase - stepStart), activeThreadsPerBlock
//    );
//
//    index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
//    output[index] = intervalsTile[threadIdx.x];
//    output[blockDim.x + index] = intervalsTile[blockDim.x + threadIdx.x];
//}
//
///*
//Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
//*/
//__global__ void bitonicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t phase, bool orderAsc) {
//    extern __shared__ el_t mergeTile[];
//    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
//    interval_t interval = intervals[blockIdx.x];
//    // Elements inside same sub-block have to be ordered in same direction
//    bool direction = orderAsc ^ ((index >> phase) & 1);
//
//    // Every thread loads 2 elements
//    mergeTile[threadIdx.x] = getTableElement(input, interval, threadIdx.x);
//    mergeTile[blockDim.x + threadIdx.x] = getTableElement(input, interval, blockDim.x + threadIdx.x);
//
//    // Bitonic merge
//    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
//        __syncthreads();
//        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
//        compareExchange(&mergeTile[start], &mergeTile[start + stride], direction);
//    }
//
//    __syncthreads();
//    output[index] = mergeTile[threadIdx.x];
//    output[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
//}
