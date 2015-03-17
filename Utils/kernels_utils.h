#ifndef KERNELS_UTILS_H
#define KERNELS_UTILS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"


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

#endif
