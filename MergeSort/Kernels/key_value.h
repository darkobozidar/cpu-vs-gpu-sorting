#ifndef KERNELS_KEY_VALUE_MERGE_SORT_H
#define KERNELS_KEY_VALUE_MERGE_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"


/*
Sorts sub blocks of input data with merge sort. Sort is stable.
*/
template <uint_t threadsMerge, uint_t elemsMergeSort, order_t sortOrder>
__global__ void mergeSortKernel(data_t *keys, data_t *values)
{
    extern __shared__ data_t mergeSortTile[];

    uint_t elemsPerThreadBlock = threadsMerge * elemsMergeSort;
    data_t *globalKeys = keys + blockIdx.x * elemsPerThreadBlock;
    data_t *globalValues = values + blockIdx.x * elemsPerThreadBlock;

    // Buffer array is needed in case every thread sorts more than 2 elements
    data_t *dataKeys = mergeSortTile;
    data_t *dataValues = mergeSortTile + elemsPerThreadBlock;
    data_t *bufferKeys = mergeSortTile + 2 * elemsPerThreadBlock;
    data_t *bufferValues = mergeSortTile + 3 * elemsPerThreadBlock;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        dataKeys[tx] = globalKeys[tx];
        dataValues[tx] = globalValues[tx];
    }

    // Stride - length of sorted blocks
    for (uint_t stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
    {
        __syncthreads();

        for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)
        {
            // Offset of current sample within block
            uint_t offsetSample = tx & (stride - 1);
            // Ofset to two blocks of length stride, which will be merged
            uint_t offsetBlock = 2 * (tx - offsetSample);

            // Loads element from even and odd block (blocks being merged)
            uint_t indexEven = offsetBlock + offsetSample;
            uint_t indexOdd = indexEven + stride;

            data_t keyEven = dataKeys[indexEven];
            data_t valueEven = dataValues[indexEven];
            data_t keyOdd = dataKeys[indexOdd];
            data_t valueOdd = dataValues[indexOdd];

            // Calculate the rank of element from even block in odd block and vice versa
            uint_t rankOdd = binarySearchInclusive<sortOrder>(
                dataKeys, keyEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
            );
            uint_t rankEven = binarySearchExclusive<sortOrder>(
                dataKeys, keyOdd, offsetBlock, offsetBlock + stride - 1
            );

            // Stores elements
            indexEven = offsetSample + rankOdd - stride;
            indexOdd = offsetSample + rankEven;

            bufferKeys[indexEven] = keyEven;
            bufferValues[indexEven] = valueEven;
            bufferKeys[indexOdd] = keyOdd;
            bufferValues[indexOdd] = valueOdd;
        }

        // Exchanges keys and values pointers with buffer pointers
        data_t *temp = dataKeys;
        dataKeys = bufferKeys;
        bufferKeys = temp;

        temp = dataValues;
        dataValues = bufferValues;
        bufferValues = temp;
    }

    __syncthreads();
    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        globalKeys[tx] = dataKeys[tx];
        globalValues[tx] = dataValues[tx];
    }
}

/*
Merges consecutive even and odd sub-blocks determined by ranks.
*/
template <uint_t subBlockSize, order_t sortOrder>
__global__ void mergeKernel(
    data_t *keys, data_t *values, data_t *keysBuffer, data_t *valuesBuffer, uint_t *ranksEven, uint_t *ranksOdd,
    uint_t sortedBlockSize
)
{
    __shared__ data_t keysEven[subBlockSize];
    __shared__ data_t keysOdd[subBlockSize];
    // Values don't need to be read in shared memory, because we need to search only in keys. Value
    // variables are used to read values in coalesced manner when keys are read from global memory.
    data_t valueEven, valueOdd;

    uint_t indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;
    uint_t indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;

    // Indexes for neighboring even and odd blocks, which will be merged
    uint_t indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
    uint_t offsetEven, offsetOdd;
    uint_t numElementsEven, numElementsOdd;

    // Reads the START index for even and odd sub-blocks
    if (blockIdx.x > 0)
    {
        indexStartEven = ranksEven[indexRank - 1];
        indexStartOdd = ranksOdd[indexRank - 1];
    }
    else
    {
        indexStartEven = 0;
        indexStartOdd = 0;
    }
    // Reads the END index for even and odd sub-blocks
    if (blockIdx.x < gridDim.x - 1)
    {
        indexEndEven = ranksEven[indexRank];
        indexEndOdd = ranksOdd[indexRank];
    }
    else
    {
        indexEndEven = sortedBlockSize;
        indexEndOdd = sortedBlockSize;
    }

    numElementsEven = indexEndEven - indexStartEven;
    numElementsOdd = indexEndOdd - indexStartOdd;

    // Reads data for sub-block in EVEN sorted block
    // Values are also read alongside keys because they are read in coalesced manner
    if (threadIdx.x < numElementsEven)
    {
        offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
        keysEven[threadIdx.x] = keys[offsetEven];
        valueEven = values[offsetEven];
    }
    // Reads data for sub-block in ODD sorted block
    if (threadIdx.x < numElementsOdd)
    {
        offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
        keysOdd[threadIdx.x] = keys[offsetOdd + sortedBlockSize];
        valueOdd = values[offsetOdd + sortedBlockSize];
    }

    __syncthreads();
    // Search for ranks in ODD sub-block for all elements in EVEN sub-block
    if (threadIdx.x < numElementsEven)
    {
        uint_t rankOdd = binarySearchInclusive<sortOrder>(keysOdd, keysEven[threadIdx.x], 0, numElementsOdd - 1);
        rankOdd += indexStartOdd;

        keysBuffer[offsetEven + rankOdd] = keysEven[threadIdx.x];
        valuesBuffer[offsetEven + rankOdd] = valueEven;
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd)
    {
        uint_t rankEven = binarySearchExclusive<sortOrder>(keysEven, keysOdd[threadIdx.x], 0, numElementsEven - 1);
        rankEven += indexStartEven;

        keysBuffer[offsetOdd + rankEven] = keysOdd[threadIdx.x];
        valuesBuffer[offsetOdd + rankEven] = valueOdd;
    }
}

#endif
