#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"
#include "data_types.h"


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
__global__ void addPaddingKernel(data_t *dataTable, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        uint_t index = offset + tx;
        dataTable[index] = value;
    }
}

template __global__ void addPaddingKernel<MIN_VAL>(
    data_t *dataTable, uint_t start, uint_t length
);
template __global__ void addPaddingKernel<MAX_VAL>(
    data_t *dataTable, uint_t start, uint_t length
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
Reads every SUB_BLOCK_SIZE-th sample from data table and orders samples, which came from the same
ordered block.
Before blocks of samples are sorted, their ranks in sorted block are saved.
*/
template <order_t sortOrder>
__global__ void generateSamplesKernel(data_t *dataTable, sample_t *samples, uint_t sortedBlockSize)
{
    // Indexes of sample in global memory and in table of samples
    uint_t dataIndex = blockIdx.x * (blockDim.x * SUB_BLOCK_SIZE) + threadIdx.x * SUB_BLOCK_SIZE;
    uint_t sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculates index of current sorted block and opposite block, with wich current block
    // will be merged (even - odd and vice versa)
    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t indexBlockCurrent = sampleIndex / subBlocksPerSortedBlock;
    uint_t indexBlockOpposite = indexBlockCurrent ^ 1;
    sample_t sample;
    uint_t rank;

    // Reads the sample and the current rank in table
    sample.value = dataTable[dataIndex];
    sample.index = sampleIndex;

    // If current sample came from even block, search in corresponding odd block (and vice versa)
    if (indexBlockCurrent % 2 == 0)
    {
        rank = binarySearchInclusive<sortOrder, SUB_BLOCK_SIZE>(
            dataTable, sample.value, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE
        );
        rank = (rank - sortedBlockSize) / SUB_BLOCK_SIZE;
    }
    else
    {
        rank = binarySearchExclusive<sortOrder, SUB_BLOCK_SIZE>(
            dataTable, sample.value, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE
        );
        rank /= SUB_BLOCK_SIZE;
    }

    samples[sampleIndex % subBlocksPerSortedBlock + rank] = sample;
}

template __global__ void generateSamplesKernel<ORDER_ASC>(
    data_t *dataTable, sample_t *samples, uint_t sortedBlockSize
);
template __global__ void generateSamplesKernel<ORDER_DESC>(
    data_t *dataTable, sample_t *samples, uint_t sortedBlockSize
);


/*
From array of sorted samples for every soted block generates the ranks/limits of sub-blocks,
which will be merged by merge kernel.
*/
template <order_t sortOrder>
__global__ void generateRanksKernel(
    data_t* dataTable, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
)
{
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Key -> sample value, Val -> rank of sample element in current table
    sample_t sample = samples[index];
    // Calculates ranks of current and opposite sorted block in global table
    uint_t rankDataCurrent = (sample.index * SUB_BLOCK_SIZE % sortedBlockSize) + 1;
    uint_t rankDataOpposite;

    // Calculates index of opposite block, with wich current block will be merged
    uint_t offsetBlockOpposite = (sample.index / subBlocksPerSortedBlock) ^ 1;
    // Calculate the index of sub-block within sorted block
    uint_t offsetSubBlockOpposite = index % subBlocksPerMergedBlock - sample.index % subBlocksPerSortedBlock - 1;
    // Start and end index for binary search
    uint_t indexStart = offsetBlockOpposite * sortedBlockSize + offsetSubBlockOpposite * SUB_BLOCK_SIZE + 1;
    uint_t indexEnd = indexStart + SUB_BLOCK_SIZE - 2;

    // Has to be explicitly converted to int, because it can be negative
    if ((int_t)(indexStart - offsetBlockOpposite * sortedBlockSize) >= 0)
    {
        if (offsetBlockOpposite % 2 == 0)
        {
            rankDataOpposite = binarySearchExclusive<sortOrder, 1>(
                dataTable, sample.value, indexStart, indexEnd
            );
        }
        else
        {
            rankDataOpposite = binarySearchInclusive<sortOrder, 1>(
                dataTable, sample.value, indexStart, indexEnd
            );
        }

        rankDataOpposite -= offsetBlockOpposite * sortedBlockSize;
    }
    else
    {
        rankDataOpposite = 0;
    }

    if ((sample.index / subBlocksPerSortedBlock) % 2 == 0)
    {
        ranksEven[index] = rankDataCurrent;
        ranksOdd[index] = rankDataOpposite;
    }
    else
    {
        ranksEven[index] = rankDataOpposite;
        ranksOdd[index] = rankDataCurrent;
    }
}

template __global__ void generateRanksKernel<ORDER_ASC>(
    data_t* dataTable, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);
template __global__ void generateRanksKernel<ORDER_DESC>(
    data_t* dataTable, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);


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
