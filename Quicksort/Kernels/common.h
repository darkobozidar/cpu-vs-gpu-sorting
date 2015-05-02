#ifndef KERNELS_COMMON_QUICKSORT_H
#define KERNELS_COMMON_QUICKSORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/constants_common.h"
#include "../../Utils/kernels_utils.h"
#include "../data_types.h"
#include "common_utils.h"


/*
From input array finds min/max value and outputs the min/max value to output.
*/
template <uint_t threadsReduction, uint_t elemsThreadReduction>
__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen)
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + threadsReduction;

    uint_t offset, dataBlockLength;
    calcDataBlockLength<threadsReduction, elemsThreadReduction>(offset, dataBlockLength, tableLen);

    data_t minVal = MAX_VAL;
    data_t maxVal = MIN_VAL;

    // Every thread reads and processes multiple elements
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsReduction)
    {
        data_t val = input[offset + tx];
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
    }

    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Once all threads have processed their corresponding elements, than reduction is done in shared memory
    if (threadIdx.x < threadsReduction / 2)
    {
        warpMinReduce<threadsReduction>(minValues);
    }
    else
    {
        warpMaxReduce<threadsReduction>(maxValues);
    }
    __syncthreads();

    // First warp loads results from all other warps and performs reduction
    if ((threadIdx.x >> WARP_SIZE_LOG) == 0)
    {
        // Every warp reduces 2 * warpSize elements
        uint_t index = threadIdx.x << (WARP_SIZE_LOG + 1);

        // Threads load results of all other warp and half of those warps performs reduction on results
        if (index < threadsReduction && threadsReduction > WARP_SIZE)
        {
            minValues[threadIdx.x] = minValues[index];
            maxValues[threadIdx.x] = maxValues[index];

            if (index < threadsReduction / 2)
            {
                warpMinReduce<(threadsReduction >> (WARP_SIZE_LOG + 1))>(minValues);
            }
            else
            {
                warpMaxReduce<(threadsReduction >> (WARP_SIZE_LOG + 1))>(maxValues);
            }
        }

        // First thread in thread block outputs reduced results
        if (threadIdx.x == 0)
        {
            output[blockIdx.x] = minValues[0];
            output[gridDim.x + blockIdx.x] = maxValues[0];
        }
    }
}

#endif
