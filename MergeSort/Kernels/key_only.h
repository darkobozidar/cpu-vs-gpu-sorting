#ifndef KERNELS_KEY_ONLY_MERGE_SORT_H
#define KERNELS_KEY_ONLY_MERGE_SORT_H

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
template <uint_t threadsMerge, uint_t elemsThreadMerge, order_t sortOrder>
__global__ void mergeSortKernel(data_t *dataTable)
{
    extern __shared__ data_t mergeSortTile[];

    uint_t elemsPerThreadBlock = threadsMerge * elemsThreadMerge;
    data_t *globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;

    // Buffer array is needed in case every thread sorts more than 2 elements
    data_t *mergeTile = mergeSortTile;
    data_t *bufferTile = mergeTile + elemsPerThreadBlock;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        mergeTile[tx] = globalDataTable[tx];
    }

    // Stride - length of sorted blocks
    for (uint_t stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
    {
        __syncthreads();

        for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)
        {
            // Offset of current sample within block
            uint_t offsetSample = tx & (stride - 1);
            // Offset to two blocks of length stride, which will be merged
            uint_t offsetBlock = 2 * (tx - offsetSample);

            // Loads element from even and odd block (blocks being merged)
            data_t elemEven = mergeTile[offsetBlock + offsetSample];
            data_t elemOdd = mergeTile[offsetBlock + offsetSample + stride];

            // Calculate the rank of element from even block in odd block and vice versa
            uint_t rankOdd = binarySearchInclusive<sortOrder>(
                mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
            );
            uint_t rankEven = binarySearchExclusive<sortOrder>(
                mergeTile, elemOdd, offsetBlock, offsetBlock + stride - 1
            );

            bufferTile[offsetSample + rankOdd - stride] = elemEven;
            bufferTile[offsetSample + rankEven] = elemOdd;
        }

        data_t *temp = mergeTile;
        mergeTile = bufferTile;
        bufferTile = temp;
    }

    __syncthreads();
    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
    {
        globalDataTable[tx] = mergeTile[tx];
    }
}

/*
Merges consecutive even and odd sub-blocks determined by ranks.
*/
template <uint_t subBlockSize, order_t sortOrder>
__global__ void mergeKernel(
    data_t *keys, data_t *keysBuffer, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
)
{
    __shared__ data_t tileEven[subBlockSize];
    __shared__ data_t tileOdd[subBlockSize];

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
    if (threadIdx.x < numElementsEven)
    {
        offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
        tileEven[threadIdx.x] = keys[offsetEven];
    }
    // Reads data for sub-block in ODD sorted block
    if (threadIdx.x < numElementsOdd)
    {
        offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
        tileOdd[threadIdx.x] = keys[offsetOdd + sortedBlockSize];
    }

    __syncthreads();
    // Search for ranks in ODD sub-block for all elements in EVEN sub-block
    if (threadIdx.x < numElementsEven)
    {
        uint_t rankOdd = binarySearchInclusive<sortOrder>(tileOdd, tileEven[threadIdx.x], 0, numElementsOdd - 1);
        rankOdd += indexStartOdd;
        keysBuffer[offsetEven + rankOdd] = tileEven[threadIdx.x];
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd)
    {
        uint_t rankEven = binarySearchExclusive<sortOrder>(tileEven, tileOdd[threadIdx.x], 0, numElementsEven - 1);
        rankEven += indexStartEven;
        keysBuffer[offsetOdd + rankEven] = tileOdd[threadIdx.x];
    }
}

#endif
