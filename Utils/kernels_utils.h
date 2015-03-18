#ifndef KERNELS_UTILS_H
#define KERNELS_UTILS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants_common.h"
#include "data_types_common.h"


/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
inline __device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if (sortOrder == ORDER_ASC ? (*elem1 > *elem2) : (*elem1 < *elem2))
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
    if (sortOrder == ORDER_ASC ? (*key1 > *key2) : (*key1 < *key2))
    {
        data_t temp = *key1;
        *key1 = *key2;
        *key2 = temp;

        temp = *val1;
        *val1 = *val2;
        *val2 = temp;
    }
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

/*
From provided number of threads in thread block, number of elements processed by one thread and array length
calculates the offset and length of data block, which is processed by current thread block.
*/
template <uint_t numThreads, uint_t elemsThread>
inline __device__ void calcDataBlockLength(uint_t &offset, uint_t &dataBlockLength, uint_t arrayLength)
{
    uint_t elemsPerThreadBlock = numThreads * elemsThread;
    offset = blockIdx.x * elemsPerThreadBlock;
    dataBlockLength =  offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;
}

/*
Binary search, which returns an index of last element LOWER than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
template <order_t sortOrder, uint_t stride>
inline __device__ int_t binarySearchExclusive(data_t* dataArray, data_t target, int_t indexStart, int_t indexEnd)
{
    while (indexStart <= indexEnd)
    {
        // Floor to multiplier of stride - needed for strides > 1
        int_t index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ MAX_VAL);

        if (sortOrder == ORDER_ASC ? (target < dataArray[index]) : (target > dataArray[index]))
        {
            indexEnd = index - stride;
        }
        else
        {
            indexStart = index + stride;
        }
    }

    return indexStart;
}

/*
Performs excluesive binary search with stride 1 (which is used in most cases).
*/
template <order_t sortOrder>
inline __device__ int_t binarySearchExclusive(data_t* dataArray, data_t target, int_t indexStart, int_t indexEnd)
{
    return binarySearchExclusive<sortOrder, 1>(dataArray, target, indexStart, indexEnd);
}

/*
Binary search, which returns an index of last element LOWER OR EQUAL than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
template <order_t sortOrder, uint_t stride>
inline __device__ int_t binarySearchInclusive(data_t* dataArray, data_t target, int_t indexStart, int_t indexEnd)
{
    while (indexStart <= indexEnd)
    {
        // Floor to multiplier of stride - needed for strides > 1
        int_t index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ MAX_VAL);

        if (sortOrder == ORDER_ASC ? (target <= dataArray[index]) : (target >= dataArray[index]))
        {
            indexEnd = index - stride;
        }
        else
        {
            indexStart = index + stride;
        }
    }

    return indexStart;
}

/*
Performs inclusive binary search with stride 1 (which is used in most cases).
*/
template <order_t sortOrder>
inline __device__ int_t binarySearchInclusive(data_t* dataArray, data_t target, int_t indexStart, int_t indexEnd)
{
    return binarySearchInclusive<sortOrder, 1>(dataArray, target, indexStart, indexEnd);
}

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

#endif
