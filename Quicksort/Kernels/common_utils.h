#ifndef KERNEL_COMMON_UTILS_QUICKSORT_H
#define KERNEL_COMMON_UTILS_QUICKSORT_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../data_types.h"
#include "../constants.h"


//////////////////////////// GENERAL UTILS //////////////////////////

/*
Calculates median of 3 provided values.
*/
inline __device__ data_t getMedian(data_t a, data_t b, data_t c)
{
    data_t maxVal = max(max(a, b), c);
    data_t minVal = min(min(a, b), c);
    return a ^ b ^ c ^ maxVal ^ minVal;
}


////////////////////////// MIN/MAX REDUCTION ////////////////////////

/*
Performs parallel min/max reduction. Half of the threads in thread block calculates min value,
other half calculates max value. Result is returned as the first element in each array.
*/
template <uint_t blockSize>
inline __device__ void minMaxReduction()
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + blockSize;

    for (uint_t stride = blockSize / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            minValues[threadIdx.x] = min(minValues[threadIdx.x], minValues[threadIdx.x + stride]);
        }
        else if (threadIdx.x < 2 * stride)
        {
            maxValues[threadIdx.x - stride] = max(maxValues[threadIdx.x - stride], maxValues[threadIdx.x]);
        }
        __syncthreads();
    }
}

/*
Min reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
inline __device__ void warpMinReduce(volatile data_t *minValues)
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
inline __device__ void warpMaxReduce(volatile data_t *maxValues) {
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


/////////////////////// GLOBAL QUICKSORT UTILS //////////////////////

/*
Counts the number of elements which are lower and greater than pivot.
*/
template <uint_t threadsSortGlobal>
__device__ void countElementsLowerGreaterPivot(
    data_t *keys, d_glob_seq_t *sequences, d_glob_seq_t sequence, uint_t seqIdx, uint_t localStart,
    uint_t localLength, uint_t &localLower, uint_t &localGreater, uint_t &scanLower, uint_t &scanGreater
)
{
    extern __shared__ data_t globalSortTile[];
#if USE_REDUCTION_IN_GLOBAL_SORT
    data_t *minValues = globalSortTile;
    data_t *maxValues = globalSortTile + threadsSortGlobal;
    // Initializes min/max values.
    data_t minVal = MAX_VAL, maxVal = MIN_VAL;
#endif

    // Counts the number of elements lower/greater than pivot and finds min/max
    for (uint_t tx = threadIdx.x; tx < localLength; tx += threadsSortGlobal)
    {
        data_t temp = keys[localStart + tx];
        localLower += temp < sequence.pivot;
        localGreater += temp > sequence.pivot;

#if USE_REDUCTION_IN_GLOBAL_SORT
        // Max value is calculated for "lower" sequence and min value is calculated for "greater" sequence.
        // Min for lower sequence and max of greater sequence (min and max of currently partitioned
        // sequence) were already calculated on host.
        maxVal = max(maxVal, temp < sequence.pivot ? temp : MIN_VAL);
        minVal = min(minVal, temp > sequence.pivot ? temp : MAX_VAL);
#endif
    }

#if USE_REDUCTION_IN_GLOBAL_SORT
    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Calculates and saves min/max values, before shared memory gets overridden by scan
    minMaxReduction<threadsSortGlobal>();
    if (threadIdx.x == (threadsSortGlobal - 1))
    {
        atomicMin(&sequences[seqIdx].greaterSeqMinVal, minValues[0]);
        atomicMax(&sequences[seqIdx].lowerSeqMaxVal, maxValues[0]);
    }
#endif
    __syncthreads();

    // Calculates number of elements lower/greater than pivot inside whole thread blocks
    scanLower = intraBlockScan<threadsSortGlobal>(localLower);
    __syncthreads();
    scanGreater = intraBlockScan<threadsSortGlobal>(localGreater);
    __syncthreads();
}


//////////////////////// LOCAL QUICKSORT UTILS //////////////////////

/*
Returns last local sequence on workstack and decreases workstack counter (pop).
*/
inline __device__ loc_seq_t popWorkstack(loc_seq_t *workstack, int_t &workstackCounter)
{
    if (threadIdx.x == 0)
    {
        workstackCounter--;
    }
    __syncthreads();

    return workstack[workstackCounter + 1];
}

/*
From provided sequence generates 2 new sequences and pushes them on stack of sequences.
*/
inline __device__ int_t pushWorkstack(
    loc_seq_t *workstack, int_t &workstackCounter, loc_seq_t sequence, data_t pivot, uint_t lowerCounter,
    uint_t greaterCounter
)
{
    loc_seq_t newSequence1, newSequence2;

    newSequence1.direction = (direct_t)!sequence.direction;
    newSequence2.direction = (direct_t)!sequence.direction;
    bool isLowerShorter = lowerCounter <= greaterCounter;

    // From provided sequence generates new sequences
    newSequence1.start = isLowerShorter ? sequence.start + sequence.length - greaterCounter : sequence.start;
    newSequence1.length = isLowerShorter ? greaterCounter : lowerCounter;
    newSequence2.start = isLowerShorter ? sequence.start : sequence.start + sequence.length - greaterCounter;
    newSequence2.length = isLowerShorter ? lowerCounter : greaterCounter;

    // Push news sequences on stack
    if (newSequence1.length > 0)
    {
        workstack[++workstackCounter] = newSequence1;
    }
    if (newSequence2.length > 0)
    {
        workstack[++workstackCounter] = newSequence2;
    }

    return workstackCounter;
}

#endif
