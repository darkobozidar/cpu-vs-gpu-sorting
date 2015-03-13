#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "data_types.h"


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

/*
Calculates the next power of 2 of provided value or returns value if it is already a power of 2.
*/
inline __device__ uint_t nextPowerOf2Device(uint_t value)
{
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;

    return value;
}


///////////////////////////// SCAN UTILS ////////////////////////////

/*
Performs exclusive scan and computes, how many elements have 'true' predicate before current element.
*/
template <uint_t blockSize>
inline __device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val)
{
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

    scanTile[index] = 0;
    index += min(blockSize, WARP_SIZE);
    scanTile[index] = val;

    if (blockSize >= 2)
    {
        scanTile[index] += scanTile[index - 1];
    }
    if (blockSize >= 4)
    {
        scanTile[index] += scanTile[index - 2];
    }
    if (blockSize >= 8)
    {
        scanTile[index] += scanTile[index - 4];
    }
    if (blockSize >= 16)
    {
        scanTile[index] += scanTile[index - 8];
    }
    if (blockSize >= 32)
    {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

/*
Performs intra-block INCLUSIVE scan.
*/
template <uint_t blockSize>
inline __device__ uint_t intraBlockScan(uint_t val)
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / WARP_SIZE;
    uint_t laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

    uint_t warpResult = intraWarpScan<blockSize>(scanTile, val);
    __syncthreads();

    if (laneIdx == WARP_SIZE - 1)
    {
        scanTile[warpIdx] = warpResult + val;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < WARP_SIZE)
    {
        scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx] + val;
}


////////////////////////// MIN/MAX REDUCTION ////////////////////////

/*
Performs parallel min/max reduction. Half of the threads in thread block calculates min value,
other half calculates max value. Result is returned as the first element in each array.
*/
template <uint_t blockSize>
inline __device__ void minMaxReduction(uint_t length)
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + blockSize;

    for (uint_t stride = length / 2; stride > 0; stride >>= 1)
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


///////////////////////// BITONIC SORT UTILS ////////////////////////

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
inline __device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
inline __device__ void compareExchange(data_t *key1, data_t *key2, data_t *val1, data_t *val2)
{
    if ((*key1 > *key2) ^ sortOrder)
    {
        data_t temp = *key1;
        *key1 = *key2;
        *key2 = temp;

        temp = *val1;
        *val1 = *val2;
        *val2 = temp;
    }
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
