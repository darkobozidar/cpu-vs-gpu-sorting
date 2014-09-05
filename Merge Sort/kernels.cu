#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


/*
Binary search, which returns an index of last element LOWER than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
__device__ int binarySearchExclusive(el_t* dataTile, el_t target, int_t indexStart, int_t indexEnd,
                                     bool orderAsc) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;

        if ((target.key < dataTile[index].key) ^ (!orderAsc)) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

/*
Binary search, which returns an index of last element LOWER OR EQUAL than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
__device__ int binarySearchInclusive(el_t* dataTile, el_t target, int_t indexStart, int_t indexEnd,
                                     bool orderAsc) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;

        if ((target.key <= dataTile[index].key) ^ (!orderAsc)) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

/*
Sorts sub blocks of input data with merge sort. Sort is stable.
*/
__global__ void mergeSortKernel(el_t *input, el_t *output, bool orderAsc) {
    __shared__ el_t tile[SHARED_MEM_SIZE];

    // Every thread loads 2 elements
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = input[index];
    tile[threadIdx.x + blockDim.x] = input[index + blockDim.x];

    // Stride - length of sorted blocks
    for (uint_t stride = 1; stride < SHARED_MEM_SIZE; stride <<= 1) {
        // Offset of current sample within block
        uint_t offsetSample = threadIdx.x & (stride - 1);
        // Ofset to two blocks of length stride, which will be merged
        uint_t offsetBlock = 2 * (threadIdx.x - offsetSample);

        __syncthreads();
        // Load element from even and odd block (blocks beeing merged)
        el_t elEven = tile[offsetBlock + offsetSample];
        el_t elOdd = tile[offsetBlock + offsetSample + stride];

        // Calculate the rank of element from even block in odd block and vice versa
        uint_t rankOdd = binarySearchInclusive(
            tile, elEven, offsetBlock + stride, offsetBlock + 2 * stride - 1, orderAsc
        );
        uint_t rankEven = binarySearchExclusive(
            tile, elOdd, offsetBlock, offsetBlock + stride - 1, orderAsc
        );

        __syncthreads();
        tile[offsetSample + rankOdd - stride] = elEven;
        tile[offsetSample + rankEven] = elOdd;
    }

    __syncthreads();
    output[index] = tile[threadIdx.x];
    output[index + blockDim.x] = tile[threadIdx.x + blockDim.x];
}

/*
Binary search, which returns an index of last element LOWER than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
__device__ int binIn(el_t* dataTile, uint_t target, int_t indexStart, int_t indexEnd,
    bool orderAsc) {

    while (indexStart <= indexEnd) {
        int index = ((indexStart + indexEnd) / 2) & ((SUB_BLOCK_SIZE - 1) ^ ULONG_MAX);

        if ((target < dataTile[index].key) ^ (!orderAsc)) {
            indexEnd = index - SUB_BLOCK_SIZE;
        }
        else {
            indexStart = index + SUB_BLOCK_SIZE;
        }
    }

    return indexStart;
}

/*
Binary search, which returns an index of last element LOWER OR EQUAL than target.
Start and end indexes can't be unsigned, because end index can become negative.
*/
__device__ int binEx(el_t* dataTile, uint_t target, int_t indexStart, int_t indexEnd,
    bool orderAsc) {

    while (indexStart <= indexEnd) {
        int index = ((indexStart + indexEnd) / 2) & ((SUB_BLOCK_SIZE - 1) ^ ULONG_MAX);

        if ((target <= dataTile[index].key) ^ (!orderAsc)) {
            indexEnd = index - SUB_BLOCK_SIZE;
        }
        else {
            indexStart = index + SUB_BLOCK_SIZE;
        }
    }

    return indexStart;
}

/*
Extracts samples from table in such a way, that they can be merged with bitonic merge.
Even samples are read in same order as they are in table, odd samples are read in reverse order.
*/
__global__ void generateSamplesKernel(el_t *table, sample_t *samples, uint_t sortedBlockSize) {
    uint_t dataIndex = blockIdx.x * (blockDim.x * SUB_BLOCK_SIZE) + threadIdx.x * SUB_BLOCK_SIZE;
    uint_t sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;
    uint_t indexBlockCurr = sampleIndex / subBlocksPerSortedBlock;
    uint_t indexBlockOpposite = indexBlockCurr ^ 1;
    sample_t sample;
    uint_t rank;

    sample.key = table[dataIndex].key;
    sample.rank = sampleIndex;

    if (indexBlockCurr % 2 == 0) {
        rank = binEx(
            table, table[dataIndex].key, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE, 1
        );
        rank = (rank - sortedBlockSize) / SUB_BLOCK_SIZE;
    } else {
        rank = binIn(
            table, table[dataIndex].key, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - SUB_BLOCK_SIZE, 1
        );
        rank /= SUB_BLOCK_SIZE;
    }

    samples[sampleIndex % subBlocksPerSortedBlock + rank] = sample;
}

/*
Binary search, which return rank in the opposite sub-block of specified sample.
*/
__device__ uint_t binarySearchRank(el_t* table, uint_t sortedBlockSize, uint_t subBlockSize,
    uint_t rank, uint_t target) {
    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Offset to current merged group of sorted blocks...
    uint_t offsetBlockOpposite = (rank / subBlocksPerMergedBlock) * 2;
    // ... + offset to odd / even block
    offsetBlockOpposite += !((rank % subBlocksPerMergedBlock) / subBlocksPerSortedBlock);
    // Calculate the rank in opposite block
    uint_t offsetSubBlockOpposite = (blockIdx.x * blockDim.x + threadIdx.x) % subBlocksPerMergedBlock - rank % subBlocksPerSortedBlock - 1;

    uint_t indexStart = offsetBlockOpposite * sortedBlockSize + offsetSubBlockOpposite * SUB_BLOCK_SIZE + 1;
    uint_t indexEnd = indexStart + SUB_BLOCK_SIZE - 2;

    // Has to be explicitly converted to int, because it can be negative
    if ((int_t)(indexStart - offsetBlockOpposite * sortedBlockSize) >= 0) {
        while (indexStart <= indexEnd) {
            uint_t index = (indexStart + indexEnd) / 2;
            el_t currEl = table[index];

            if (target <= currEl.key) {
                indexEnd = index - 1;
            }
            else {
                indexStart = index + 1;
            }
        }

        return indexStart - offsetBlockOpposite * sortedBlockSize;
    }

    return 0;
}

__global__ void generateRanksKernel(el_t* table, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd,
    uint_t tableLen, uint_t sortedBlockSize) {
    uint_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Calculate ranks of current and opposite sorted block in global table
    sample_t sample = samples[index];
    uint_t key = samples[index].key;
    uint_t rank = samples[index].rank;
    uint_t rankDataCurrent = (rank * SUB_BLOCK_SIZE % sortedBlockSize) + 1;
    uint_t rankDataOpposite = binarySearchRank(table, sortedBlockSize, SUB_BLOCK_SIZE, rank, key);

    if ((rank / subBlocksPerSortedBlock) % 2 == 0) {
        ranksEven[index] = rankDataCurrent;
        ranksOdd[index] = rankDataOpposite;
    }
    else {
        ranksEven[index] = rankDataOpposite;
        ranksOdd[index] = rankDataCurrent;
    }

    /*__syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 64; i++) {
            printf("%2d, ", table[i].key);
        }
        printf("\n");

        for (int i = 0; i < 16; i++) {
            printf("(%2d, %2d), ", samples[i]);
        }
        printf("\n\n");
    }*/

    /*__syncthreads();
    printf("%2d %2d: %2d\n", threadIdx.x, key, ranksEven[blockIdx.x * blockDim.x + threadIdx.x]);
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("\n");
    }
    printf("%2d %2d: %2d\n", threadIdx.x, key, ranksOdd[blockIdx.x * blockDim.x + threadIdx.x]);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("\n\n");
    }*/
}

__global__ void mergeKernel(el_t* input, el_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t tableLen,
                            uint_t sortedBlockSize, uint_t subBlockSize) {
    __shared__ el_t dataTile[2 * SUB_BLOCK_SIZE];
    uint_t indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;
    uint_t indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;
    uint_t indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
    uint_t offsetEven, offsetOdd;
    uint_t numElementsEven, numElementsOdd;

    /*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            printf("%2d, ", ranksEven[i]);
        }
        printf("\n\n");

        for (int i = 0; i < 8; i++) {
            printf("%2d, ", ranksOdd[i]);
        }
        printf("\n\n");
    }*/

    // Read the START index for even and odd sub-blocks
    if (blockIdx.x > 0) {
        indexStartEven = ranksEven[indexRank - 1];
        indexStartOdd = ranksOdd[indexRank - 1];
    } else {
        indexStartEven = 0;
        indexStartOdd = 0;
    }
    // Read the END index for even and odd sub-blocks
    if (blockIdx.x < gridDim.x - 1) {
        indexEndEven = ranksEven[indexRank];
        indexEndOdd = ranksOdd[indexRank];
    } else {
        indexEndEven = sortedBlockSize;
        indexEndOdd = sortedBlockSize;
    }

    numElementsEven = indexEndEven - indexStartEven;
    numElementsOdd = indexEndOdd - indexStartOdd;

    // Read data for sub-block in EVEN sorted block
    if (threadIdx.x < numElementsEven) {
        offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
        dataTile[threadIdx.x] = input[offsetEven];
    }
    // Read data for sub-block in ODD sorted block
    if (threadIdx.x < numElementsOdd) {
        offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
        dataTile[subBlockSize + threadIdx.x] = input[offsetOdd + sortedBlockSize];
    }

    __syncthreads();
    // Search for ranks in ODD sub-block for all elements in EVEN sub-block
    if (threadIdx.x < numElementsEven) {
        uint_t rankOdd = binarySearchInclusive(dataTile, dataTile[threadIdx.x], subBlockSize,
                                               subBlockSize + numElementsOdd - 1, 1);
        rankOdd = rankOdd - subBlockSize + indexStartOdd;
        output[offsetEven + rankOdd] = dataTile[threadIdx.x];
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd) {
        uint_t rankEven = binarySearchExclusive(dataTile, dataTile[subBlockSize + threadIdx.x],
                                                0, numElementsEven - 1, 1);
        rankEven += indexStartEven;
        output[offsetOdd + rankEven] = dataTile[subBlockSize + threadIdx.x];
    }
}
