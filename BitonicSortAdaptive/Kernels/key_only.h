#ifndef KERNELS_KEY_ONLY_BITONIC_SORT_ADAPTIVE_H
#define KERNELS_KEY_ONLY_BITONIC_SORT_ADAPTIVE_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"
#include "../data_types.h"
#include "common_utils.h"


/*
Sorts sub-blocks of input data with REGULAR bitonic sort (not NORMALIZED bitonic sort).
*/
template <uint_t threadsBitonicSort, uint_t elemsBitonicSort, order_t sortOrder>
__global__ void bitonicSortRegularKernel(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t sortTile[];
    uint_t offset, dataBlockLength;
    calcDataBlockLength<threadsBitonicSort, elemsBitonicSort>(offset, dataBlockLength, tableLen);

    // If shared memory size is lower than table length, than adjacent blocks have to be ordered in opposite
    // direction in order to create bitonic sequences.
    bool blockDirection = (sortOrder == ORDER_ASC) ^ (blockIdx.x & 1);

    // Loads data into shared memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        sortTile[tx] = dataTable[offset + tx];
    }

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            __syncthreads();
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += threadsBitonicSort)
            {
                bool direction = blockDirection ^ ((tx & subBlockSize) != 0);
                uint_t index = 2 * tx - (tx & (stride - 1));

                if (direction)
                {
                    compareExchange<ORDER_ASC>(&sortTile[index], &sortTile[index + stride]);
                }
                else
                {
                    compareExchange<ORDER_DESC>(&sortTile[index], &sortTile[index + stride]);
                }
            }
        }
    }

    // Stores sorted elements from shared to global memory
    __syncthreads();
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        dataTable[offset + tx] = sortTile[tx];
    }
}

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory size.
Executes regular bitonic merge (not normalized merge). Reads data from provided intervals.
*/
template <uint_t threadsMerge, uint_t elemsMerge, order_t sortOrder>
__global__ void bitonicMergeIntervalsKernel(data_t *keys, data_t *keysBuffer, interval_t *intervals, uint_t phase)
{
    extern __shared__ data_t mergeTile[];
    interval_t interval = intervals[blockIdx.x];

    // Elements inside same sub-block have to be ordered in same direction
    uint_t elemsPerThreadBlock = threadsMerge * elemsMerge;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    bool orderAsc = (sortOrder == ORDER_ASC) ^ ((offset >> phase) & 1);

    // Loads data from global to shared memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        mergeTile[tx] = getArrayKey(keys, interval, tx);
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = elemsPerThreadBlock / 2; stride > 0; stride >>= 1)
    {
        for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock / 2; tx += threadsMerge)
        {
            uint_t index = 2 * tx - (tx & (stride - 1));

            if (orderAsc)
            {
                compareExchange<ORDER_ASC>(&mergeTile[index], &mergeTile[index + stride]);
            }
            else
            {
                compareExchange<ORDER_DESC>(&mergeTile[index], &mergeTile[index + stride]);
            }
        }
        __syncthreads();
    }

    // Stores sorted data to buffer array
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        keysBuffer[offset + tx] = mergeTile[tx];
    }
}

#endif
