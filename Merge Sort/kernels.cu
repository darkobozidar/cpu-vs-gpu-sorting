#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, bool orderAsc) {
    if ((elem1->key <= elem2->key) ^ orderAsc) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
For debugging purposes only specified thread prints to console.
*/
__device__ void printOnce(char* text, uint_t threadIndex) {
    if (threadIdx.x == threadIndex) {
        printf(text);
    }
}

/*
For debugging purposes only thread 0 prints to console.
*/
__device__ void printOnce(char* text) {
    printOnce(text, 0);
}

/*
Sorts sub blocks of input data with bitonic sort.
*/
__global__ void bitonicSortKernel(el_t *input, el_t *output, bool orderAsc) {
    __shared__ el_t sortTile[SHARED_MEM_SIZE];

    // Every thread loads 2 elements
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sortTile[threadIdx.x] = input[index];
    sortTile[blockDim.x + threadIdx.x] = input[blockDim.x + index];

    // First log2(sortedBlockSize) - 1 phases of bitonic merge
    for (uint_t size = 2; size < 2 * blockDim.x; size <<= 1) {
        uint_t direction = (!orderAsc) ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint_t stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareExchange(&sortTile[pos], &sortTile[pos + stride], direction);
        }
    }

    // Last phase of bitonic merge
    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&sortTile[pos], &sortTile[pos + stride], orderAsc);
    }

    __syncthreads();
    output[index] = sortTile[threadIdx.x];
    output[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}

/*
Binary search, which return rank in the opposite sub-block of specified sample.
*/
__device__ uint_t binarySearchRank(el_t* table, uint_t sortedBlockSize, uint_t subBlockSize,
                                   uint_t rank, el_t targetEl) {
    uint_t subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Offset to current merged group of sorted blocks...
    uint_t offsetBlockOpposite = (rank / subBlocksPerMergedBlock) * 2;
    // ... + offset to odd / even block
    offsetBlockOpposite += !((rank % subBlocksPerMergedBlock) / subBlocksPerSortedBlock);
    // Calculate the rank in opposite block
    uint_t offsetSubBlockOpposite = threadIdx.x % subBlocksPerMergedBlock - rank % subBlocksPerSortedBlock - 1;

    uint_t indexStart = offsetBlockOpposite * sortedBlockSize + offsetSubBlockOpposite * subBlockSize + 1;
    uint_t indexEnd = indexStart + subBlockSize - 2;

    uint_t iStart = indexStart;
    uint_t iEnd = indexEnd;

    // Has to be explicitly converted to int, because it can be negative
    if ((int_t)(indexStart - offsetBlockOpposite * sortedBlockSize) >= 0) {
        while (indexStart <= indexEnd) {
            uint_t index = (indexStart + indexEnd) / 2;
            el_t currEl = table[index];

            if (targetEl.key <= currEl.key) {
                indexEnd = index - 1;
            } else {
                indexStart = index + 1;
            }
        }

        return indexStart - offsetBlockOpposite * sortedBlockSize;
    }

    return 0;
}

__global__ void generateRanksKernel(el_t* table, uint_t* ranks, uint_t dataLen, uint_t sortedBlockSize) {
    __shared__ rank_el_t ranksTile[SHARED_MEM_SIZE];

    uint_t subBlocksPerSortedBlock = sortedBlockSize / SUB_BLOCK_SIZE;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;
    uint_t indexSortedBlock = threadIdx.x / subBlocksPerSortedBlock;

    uint_t dataIndex = blockIdx.x * (blockDim.x * SUB_BLOCK_SIZE) + threadIdx.x * SUB_BLOCK_SIZE;
    // Offset to correct sorted block
    uint_t tileIndex = indexSortedBlock * subBlocksPerSortedBlock;
    // Offset for sub-block index inside block for ODD block
    tileIndex += ((indexSortedBlock % 2 == 0) * threadIdx.x) % subBlocksPerSortedBlock;
    // Offset for sub-block index inside block for EVEN block (index has to be reversed)
    tileIndex += ((indexSortedBlock % 2 == 1) * (subBlocksPerSortedBlock - (threadIdx.x + 1)))
                  % subBlocksPerSortedBlock;

    // Read the samples from global memory in to shared memory in such a way, to get a bitonic
    // sequence of samples
    if (dataIndex < dataLen) {
        ranksTile[tileIndex].el = table[dataIndex];
        ranksTile[threadIdx.x].rank = tileIndex;
    }

    for (uint_t stride = subBlocksPerSortedBlock; stride > 0; stride /= 2) {
        __syncthreads();
        // Only half of the threads have to perform bitonic merge
        if (threadIdx.x >= blockDim.x / 2) {
            continue;
        }

        uint_t sampleIndex = (2 * threadIdx.x - (threadIdx.x & (stride - 1)));
        rank_el_t left = ranksTile[sampleIndex];
        rank_el_t right = ranksTile[sampleIndex + stride];

        if (left.el.key > right.el.key || left.el.key == right.el.key && left.rank > right.rank) {
            ranksTile[sampleIndex] = right;
            ranksTile[sampleIndex + stride] = left;
        }
    }

    // Calculate ranks of current and opposite sorted block in global table
    __syncthreads();
    el_t element = ranksTile[threadIdx.x].el;
    uint_t rank = ranksTile[threadIdx.x].rank;
    uint_t rankDataCurrent = (rank * SUB_BLOCK_SIZE % sortedBlockSize) + 1;
    uint_t rankDataOpposite = binarySearchRank(table, sortedBlockSize, SUB_BLOCK_SIZE, rank, element);

    // Check if rank came from odd or even sorted block
    uint_t oddEvenOffset = (rank / subBlocksPerSortedBlock) % 2;
    // Write ranks, which came from EVEN sorted blocks in FIRST half of table of ranks and
    // write ranks, which came from ODD sorted blocks to SECOND half of ranks table
    ranks[threadIdx.x + oddEvenOffset * blockDim.x] = rankDataCurrent;
    ranks[threadIdx.x + (!oddEvenOffset) * blockDim.x] = rankDataOpposite;

    /*__syncthreads();
    printf("%2d %2d: %2d %d\n", threadIdx.x, element.key, ranks[threadIdx.x], oddEvenOffset);
    __syncthreads();
    printOnce("\n");
    printf("%2d %2d: %2d %d\n", threadIdx.x, element.key, ranks[threadIdx.x + blockDim.x], oddEvenOffset);
    printOnce("\n\n");*/
}

__device__ int binarySearchEven(el_t* dataTile, int indexStart, int indexEnd, el_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;

        if (target.key < dataTile[index].key) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__device__ int binarySearchOdd(el_t* dataTile, int indexStart, int indexEnd, el_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;

        if (target.key <= dataTile[index].key) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__global__ void mergeKernel(el_t* input, el_t* output, uint_t *ranks, uint_t tableLen,
                            uint_t ranksLen, uint_t sortedBlockSize, uint_t subBlockSize) {
    __shared__ el_t dataTile[2 * SUB_BLOCK_SIZE];
    uint_t indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;
    uint_t indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;
    uint_t indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
    uint_t offsetEven, offsetOdd;
    uint_t numElementsEven, numElementsOdd;

    // Read the START index for even and odd sub-blocks
    if (blockIdx.x > 0) {
        indexStartEven = ranks[indexRank - 1];
        indexStartOdd = ranks[(indexRank - 1) + ranksLen / 2];
    } else {
        indexStartEven = 0;
        indexStartOdd = 0;
    }
    // Read the END index for even and odd sub-blocks
    if (blockIdx.x < gridDim.x - 1) {
        indexEndEven = ranks[indexRank];
        indexEndOdd = ranks[indexRank + ranksLen / 2];
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
        uint_t rankOdd = binarySearchOdd(dataTile, subBlockSize, subBlockSize + numElementsOdd - 1,
                                         dataTile[threadIdx.x]);
        rankOdd = rankOdd - subBlockSize + indexStartOdd;
        output[offsetEven + rankOdd] = dataTile[threadIdx.x];
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd) {
        uint_t rankEven = binarySearchEven(dataTile, 0, numElementsEven - 1,
                                           dataTile[subBlockSize + threadIdx.x]);
        rankEven += indexStartEven;
        output[offsetOdd + rankEven] = dataTile[subBlockSize + threadIdx.x];
    }
}
