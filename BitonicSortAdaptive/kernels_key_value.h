#ifndef KERNELS_KEY_VALUE_BITONIC_SORT_ADAPTIVE_H
#define KERNELS_KEY_VALUE_BITONIC_SORT_ADAPTIVE_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/kernels_utils.h"
#include "data_types.h"
#include "kernels_common_utils.h"

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
Executes regular bitonic merge (not normalized merge). Reads data from provided intervals.
*/
template <uint_t threadsMerge, uint_t elemsMerge, order_t sortOrder>
__global__ void bitonicMergeIntervalsKernel(
    data_t *keys, data_t *values, data_t *keysBuffer, data_t *valuesBuffer, interval_t *intervals, uint_t phase
)
{
    extern __shared__ data_t mergeTile[];
    interval_t interval = intervals[blockIdx.x];

    // Elements inside same sub-block have to be ordered in same direction
    uint_t elemsPerThreadBlock = threadsMerge * elemsMerge;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    bool orderAsc = !sortOrder ^ ((offset >> phase) & 1);

    data_t *keysTile = mergeTile;
    data_t *valuesTile = mergeTile + elemsPerThreadBlock;

    // Loads data from global to shared memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        getArrayKeyValue(keys, values, interval, tx, &keysTile[tx], &valuesTile[tx]);
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
                compareExchange<ORDER_ASC>(
                    &keysTile[index], &keysTile[index + stride], &valuesTile[index], &valuesTile[index + stride]
                );
            }
            else
            {
                compareExchange<ORDER_DESC>(
                    &keysTile[index], &keysTile[index + stride], &valuesTile[index], &valuesTile[index + stride]
                );
            }
        }
        __syncthreads();
    }

    // Stores sorted data to buffer array
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        uint_t index = offset + tx;
        keysBuffer[index] = keysTile[tx];
        valuesBuffer[index] = valuesTile[tx];
    }
}

#endif
