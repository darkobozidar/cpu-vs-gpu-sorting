#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"


///////////////////////////////////////////////////////////////////
////////////////////////////// UTILS //////////////////////////////
///////////////////////////////////////////////////////////////////

/*
Binary search, which returns an index of last element LOWER than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
template <order_t sortOrder, uint_t stride>
__device__ int_t binarySearchExclusive(
    data_t* table, data_t target, int_t indexStart, int_t indexEnd
)
{
    while (indexStart <= indexEnd)
    {
        // Floor to multiplier of stride - needed for strides > 1
        int_t index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ MAX_VAL);

        if ((target < table[index]) ^ sortOrder)
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
Binary search, which returns an index of last element LOWER OR EQUAL than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
template <order_t sortOrder, uint_t stride>
__device__ int_t binarySearchInclusive(
    data_t* table, data_t target, int_t indexStart, int_t indexEnd
)
{
    while (indexStart <= indexEnd)
    {
        // Floor to multiplier of stride - needed for strides > 1
        int_t index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ MAX_VAL);

        if ((target <= table[index]) ^ sortOrder)
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


///////////////////////////////////////////////////////////////////
///////////////////////////// KERNELS /////////////////////////////
///////////////////////////////////////////////////////////////////


/*
Adds the padding to table from start index (original table length, which is not power of 2) to the end of the
extended array (which is the next power of 2 of the original table length).
*/
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        uint_t index = offset + tx;
        dataTable[index] = value;
        dataBuffer[index] = value;
    }
}

template __global__ void addPaddingKernel<MIN_VAL>(
    data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length
);
template __global__ void addPaddingKernel<MAX_VAL>(
    data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length
);


/*
Sorts sub blocks of input data with merge sort. Sort is stable.
*/
template <order_t sortOrder>
__global__ void mergeSortKernel(data_t *dataTable)
{
    extern __shared__ data_t mergeSortTile[];

    // Var blockDim.x needed in case there array contains less elements than one thread block in
    // this kernel can sort
    uint_t elemsPerThreadBlock = blockDim.x * ELEMS_PER_THREAD_MERGE_SORT;
    uint_t *globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;

    // Buffer array is needed in case every thread sorts more than 2 elements
    data_t *mergeTile = mergeSortTile;
    data_t *bufferTile = mergeTile + elemsPerThreadBlock;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        mergeTile[tx] = globalDataTable[tx];
    }

    // Stride - length of sorted blocks
    for (uint_t stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
    {
        __syncthreads();

        for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += blockDim.x)
        {
            // Offset of current sample within block
            uint_t offsetSample = tx & (stride - 1);
            // Ofset to two blocks of length stride, which will be merged
            uint_t offsetBlock = 2 * (tx - offsetSample);

            // Loads element from even and odd block (blocks beeing merged)
            data_t elemEven = mergeTile[offsetBlock + offsetSample];
            data_t elemOdd = mergeTile[offsetBlock + offsetSample + stride];

            // Calculate the rank of element from even block in odd block and vice versa
            uint_t rankOdd = binarySearchInclusive<sortOrder, 1>(
                mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
            );
            uint_t rankEven = binarySearchExclusive<sortOrder, 1>(
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
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += blockDim.x)
    {
        globalDataTable[tx] = mergeTile[tx];
    }
}

template __global__ void mergeSortKernel<ORDER_ASC>(data_t *dataTable);
template __global__ void mergeSortKernel<ORDER_DESC>(data_t *dataTable);


/*
Generates array of ranks/boundaries of sub-block, which will be merged.
*/
template <order_t sortOrder>
__global__ void generateRanksKernel(data_t *dataTable, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize)
{
    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Reads sample value and calculates sample's global rank
    data_t sampleValue = dataTable[blockIdx.x * (blockDim.x * SUB_BLOCK_SIZE) + threadIdx.x * SUB_BLOCK_SIZE];
    uint_t rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t rankSampleOpposite;

    // Calculates index of current sorted block and opposite sorted block, with wich current block will be
    // merged (even - odd and vice versa)
    uint_t indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;
    uint_t indexBlockOpposite = indexBlockCurrent ^ 1;

    // Searches for sample's rank in opposite block in order to calculate sample's index in merged block.
    // If current sample came from even block, it searches in corresponding odd block (and vice versa)
    if (indexBlockCurrent % 2 == 0)
    {
        rankSampleOpposite = binarySearchInclusive<sortOrder, SUB_BLOCK_SIZE>(
            dataTable, sampleValue, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE
        );
        rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / SUB_BLOCK_SIZE;
    }
    else
    {
        rankSampleOpposite = binarySearchExclusive<sortOrder, SUB_BLOCK_SIZE>(
            dataTable, sampleValue, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE
        );
        rankSampleOpposite /= SUB_BLOCK_SIZE;
    }

    // Calculates index of sample inside merged block
    uint_t sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;

    // Calculates sample's rank in current and opposite sorted block
    uint_t rankDataCurrent = (rankSampleCurrent * SUB_BLOCK_SIZE % sortedBlockSize) + 1;
    uint_t rankDataOpposite;

    // Calculate the index of sub-block within opposite sorted block
    uint_t indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
    // Start and end index for binary search
    uint_t indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * SUB_BLOCK_SIZE + 1;
    uint_t indexEnd = indexStart + SUB_BLOCK_SIZE - 2;

    // Searches for sample's index in opposite sub-block (which is inside opposite sorted block)
    // Has to be explicitly converted to int, because it can be negative
    if ((int_t)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
    {
        if (indexBlockOpposite % 2 == 0)
        {
            rankDataOpposite = binarySearchExclusive<sortOrder, 1>(
                dataTable, sampleValue, indexStart, indexEnd
            );
        }
        else
        {
            rankDataOpposite = binarySearchInclusive<sortOrder, 1>(
                dataTable, sampleValue, indexStart, indexEnd
            );
        }

        rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
    }
    else
    {
        rankDataOpposite = 0;
    }

    // Outputs ranks
    if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)
    {
        ranksEven[sortedIndex] = rankDataCurrent;
        ranksOdd[sortedIndex] = rankDataOpposite;
    }
    else
    {
        ranksEven[sortedIndex] = rankDataOpposite;
        ranksOdd[sortedIndex] = rankDataCurrent;
    }
}

template __global__ void generateRanksKernel<ORDER_ASC>(
    data_t *dataTable, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);
template __global__ void generateRanksKernel<ORDER_DESC>(
    data_t *dataTable, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);

/*
Merges consecutive even and odd sub-blocks determined by ranks.
*/
template <order_t sortOrder>
__global__ void mergeKernel(
    data_t* input, data_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
)
{
    __shared__ data_t tileEven[SUB_BLOCK_SIZE];
    __shared__ data_t tileOdd[SUB_BLOCK_SIZE];

    uint_t indexRank = blockIdx.y * (sortedBlockSize / SUB_BLOCK_SIZE * 2) + blockIdx.x;
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
        tileEven[threadIdx.x] = input[offsetEven];
    }
    // Reads data for sub-block in ODD sorted block
    if (threadIdx.x < numElementsOdd)
    {
        offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
        tileOdd[threadIdx.x] = input[offsetOdd + sortedBlockSize];
    }

    __syncthreads();
    // Search for ranks in ODD sub-block for all elements in EVEN sub-block
    if (threadIdx.x < numElementsEven)
    {
        uint_t rankOdd = binarySearchInclusive<sortOrder, 1>(
            tileOdd, tileEven[threadIdx.x], 0, numElementsOdd - 1
        );
        rankOdd += indexStartOdd;
        output[offsetEven + rankOdd] = tileEven[threadIdx.x];
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd)
    {
        uint_t rankEven = binarySearchExclusive<sortOrder, 1>(
            tileEven, tileOdd[threadIdx.x], 0, numElementsEven - 1
        );
        rankEven += indexStartEven;
        output[offsetOdd + rankEven] = tileOdd[threadIdx.x];
    }
}

template __global__ void mergeKernel<ORDER_ASC>(
    data_t* input, data_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);
template __global__ void mergeKernel<ORDER_DESC>(
    data_t* input, data_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);
