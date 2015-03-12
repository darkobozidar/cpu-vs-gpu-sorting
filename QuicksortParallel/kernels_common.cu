#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "../Utils/constants_common.h"
#include "constants.h"
#include "data_types.h"


////////////////////////// MIN/MAX REDUCTION ////////////////////////

/*
Min reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
__device__ void warpMinReduce(volatile data_t *minValues)
{
    uint_t index = (threadIdx.x >> WARP_SIZE_LOG << (WARP_SIZE_LOG + 1)) + (threadIdx.x & (WARP_SIZE - 1));

    if (blockSize >= 64)
    {
        minValues[index] = min(minValues[index], minValues[index + 32]);
    }
    if (blockSize >= 32)
    {
        minValues[index] = min(minValues[index], minValues[index + 16]);
    }
    if (blockSize >= 16)
    {
        minValues[index] = min(minValues[index], minValues[index + 8]);
    }
    if (blockSize >= 8)
    {
        minValues[index] = min(minValues[index], minValues[index + 4]);
    }
    if (blockSize >= 4)
    {
        minValues[index] = min(minValues[index], minValues[index + 2]);
    }
    if (blockSize >= 2)
    {
        minValues[index] = min(minValues[index], minValues[index + 1]);
    }
}

/*
Max reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
__device__ void warpMaxReduce(volatile data_t *maxValues) {
    uint_t tx = threadIdx.x - blockSize / 2;
    uint_t index = (tx >> WARP_SIZE_LOG << (WARP_SIZE_LOG + 1)) + (tx & (WARP_SIZE - 1));

    if (blockSize >= 64)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 32]);
    }
    if (blockSize >= 32)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 16]);
    }
    if (blockSize >= 16)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 8]);
    }
    if (blockSize >= 8)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 4]);
    }
    if (blockSize >= 4)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 2]);
    }
    if (blockSize >= 2)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 1]);
    }
}


/////////////////////////////////////////////////////////////////////
/////////////////////////////// KERNELS /////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
From input array finds min/max value and outputs the min/max value to output.
*/
template <uint_t threadsReduction, uint_t elemsThreadReduction>
__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen)
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + threadsReduction;

    uint_t elemsPerBlock = threadsReduction * elemsThreadReduction;
    uint_t offset = blockIdx.x * elemsPerBlock;
    uint_t dataBlockLength = offset + elemsPerBlock <= tableLen ? elemsPerBlock : tableLen - offset;

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

    // First warp loads results from all othwer warps and performs reduction
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

template __global__ void minMaxReductionKernel<THREADS_PER_REDUCTION_KO, ELEMENTS_PER_THREAD_REDUCTION_KO>(
    data_t *input, data_t *output, uint_t tableLen
);
template __global__ void minMaxReductionKernel<THREADS_PER_REDUCTION_KV, ELEMENTS_PER_THREAD_REDUCTION_KV>(
    data_t *input, data_t *output, uint_t tableLen
);
