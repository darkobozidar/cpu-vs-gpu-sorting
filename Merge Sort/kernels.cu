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
Compare function for sort.
*/
__host__ __device__ int_t compare(const void* elem1, const void* elem2) {
    return (*(data_t*)elem1 - *(data_t*)elem2);
}

/*
Compares 2 elements with compare function and exchanges them according to orderAsc.
*/
__host__ __device__ void compareExchange(data_t* elem1, data_t* elem2, bool orderAsc) {
    if ((compare(elem1, elem2) < 0) ^ orderAsc) {
        data_t temp = *elem1;
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
Sorts sub blocks of size sortedBlockSize with bitonic sort.
*/
__global__ void bitonicSortKernel(data_t* data, uint_t dataLen, uint_t sortedBlockSize, bool orderAsc) {
    extern __shared__ data_t sortTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    if (index < dataLen) {
        sortTile[threadIdx.x] = data[index];
    }
    if (index + blockDim.x < dataLen) {
        sortTile[threadIdx.x + blockDim.x] = data[index + blockDim.x];
    }

    // First log2(sortedBlockSize) - 1 phases of bitonic merge
    for (uint_t size = 2; size < sortedBlockSize; size <<= 1) {
        uint_t direction = (!orderAsc) ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint_t stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareExchange(&sortTile[pos], &sortTile[pos + stride], direction);
        }
    }

    // Last phase of bitonic merge
    for (uint_t stride = sortedBlockSize / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&sortTile[pos], &sortTile[pos + stride], orderAsc);
    }

    __syncthreads();
    if (index < dataLen) {
        data[index] = sortTile[threadIdx.x];
    }
    if (index + blockDim.x < dataLen) {
        data[index + blockDim.x] = sortTile[threadIdx.x + blockDim.x];
    }
}

/*
Binary search, which return rank in the opposite sub-block of specified sample.
*/
__device__ uint_t binarySearchRank(data_t* data, uint_t sortedBlockSize, uint_t subBlockSize,
                                   uint_t rank, uint_t sample) {
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
            data_t currSample = data[index];

            if (sample <= currSample) {
                indexEnd = index - 1;
            } else {
                indexStart = index + 1;
            }
        }

        return indexStart - offsetBlockOpposite * sortedBlockSize;
    }

    return 0;
}

__global__ void generateRanksKernel(data_t* data, uint_t* ranks, uint_t dataLen, uint_t sortedBlockSize,
                                    uint_t subBlockSize) {
    extern __shared__ sample_el_t ranksTile[];

    uint_t subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;
    uint_t indexSortedBlock = threadIdx.x / subBlocksPerSortedBlock;

    uint_t dataIndex = blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize;
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
        ranksTile[tileIndex].sample = data[dataIndex];
        ranksTile[threadIdx.x].rank = tileIndex;
    }
    __syncthreads();

    // TODO test on bigger tables
    for (uint_t stride = subBlocksPerSortedBlock; stride > 0; stride /= 2) {
        __syncthreads();
        // Only half of the threads have to perform bitonic merge
        if (threadIdx.x >= blockDim.x / 2) {
            continue;
        }

        uint_t sampleIndex = (2 * threadIdx.x - (threadIdx.x & (stride - 1)));

        if (ranksTile[sampleIndex].sample > ranksTile[sampleIndex + stride].sample) {
            sample_el_t temp = ranksTile[sampleIndex];
            ranksTile[sampleIndex] = ranksTile[sampleIndex + stride];
            ranksTile[sampleIndex + stride] = temp;
        }
        else if (ranksTile[sampleIndex].sample == ranksTile[sampleIndex + stride].sample && ranksTile[sampleIndex].rank > ranksTile[sampleIndex + stride].rank) {
            sample_el_t temp = ranksTile[sampleIndex];
            ranksTile[sampleIndex] = ranksTile[sampleIndex + stride];
            ranksTile[sampleIndex + stride] = temp;
        }
    }

    // Calculate ranks of current and opposite sorted block in global table
    __syncthreads();
    uint_t rank = ranksTile[threadIdx.x].rank;
    uint_t sample = ranksTile[threadIdx.x].sample;
    uint_t rankDataCurrent = (rank * subBlockSize % sortedBlockSize) + 1;
    uint_t rankDataOpposite = binarySearchRank(data, sortedBlockSize, subBlockSize, rank, sample);

    // Check if rank came from odd or even sorted block
    uint_t oddEvenOffset = (rank / subBlocksPerSortedBlock) % 2;
    // Write ranks, which came from EVEN sorted blocks in FIRST half of table of ranks and
    // write ranks, which came from ODD sorted blocks to SECOND half of ranks table
    ranks[threadIdx.x + oddEvenOffset * blockDim.x] = rankDataCurrent;
    ranks[threadIdx.x + (!oddEvenOffset) * blockDim.x] = rankDataOpposite;

    /*__syncthreads();
    printf("%2d %2d: %2d %d\n", threadIdx.x, sample, ranks[threadIdx.x], oddEvenOffset);
    __syncthreads();
    printOnce("\n");
    printf("%2d %2d: %2d %d\n", threadIdx.x, sample, ranks[threadIdx.x + blockDim.x], oddEvenOffset);
    printOnce("\n\n");*/
}

__global__ void printRanks(uint_t* ranks, uint_t ranksLen) {
    for (int i = 0; i < ranksLen; i++) {
        printf("%d, ", ranks[i]);
    }
}

__device__ int binarySearchEven(data_t* dataTile, int indexStart, int indexEnd, uint_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;
        data_t currSample = dataTile[index];

        if (target < currSample) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__device__ int binarySearchOdd(data_t* dataTile, int indexStart, int indexEnd, uint_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;
        data_t currSample = dataTile[index];

        if (target <= currSample) {
            indexEnd = index - 1;
        } else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__global__ void mergeKernel(data_t* inputData, data_t* outputData, uint_t* ranks, uint_t dataLen,
                            uint_t ranksLen, uint_t sortedBlockSize, uint_t subBlockSize) {
    extern __shared__ data_t dataTile[];
    uint_t indexRank = blockIdx.y * (sortedBlockSize / 2) + blockIdx.x;
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
        dataTile[threadIdx.x] = inputData[offsetEven];
    }
    // Read data for sub-block in ODD sorted block
    if (threadIdx.x < numElementsOdd) {
        offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
        dataTile[subBlockSize + threadIdx.x] = inputData[offsetOdd + sortedBlockSize];
    }

    __syncthreads();
    // Search for ranks in ODD sub-block for all elements in EVEN sub-block
    if (threadIdx.x < numElementsEven) {
        uint_t rankOdd = binarySearchOdd(dataTile, subBlockSize, subBlockSize + numElementsOdd - 1,
                                         dataTile[threadIdx.x]);
        rankOdd = rankOdd - subBlockSize + indexStartOdd;
        outputData[offsetEven + rankOdd] = dataTile[threadIdx.x];
    }
    // Search for ranks in EVEN sub-block for all elements in ODD sub-block
    if (threadIdx.x < numElementsOdd) {
        uint_t rankEven = binarySearchEven(dataTile, 0, numElementsEven - 1,
                                           dataTile[subBlockSize + threadIdx.x]);
        rankEven += indexStartEven;
        outputData[offsetOdd + rankEven] = dataTile[subBlockSize + threadIdx.x];
    }
}

/*if (blockIdx.x == 6 && blockIdx.y == 0 && threadIdx.x == 0) {
if (gridDim.x >= 64 && threadIdx.x == 0) {
printf("\n(%u, %u), (%u, %u)\n", indexStartEven, indexEndEven, indexStartOdd, indexEndOdd);
}
}*/

/*if (blockIdx.x == 0 && blockIdx.y == 0) {
printf("thread: %d\n", threadIdx.x);
printf("Search Interval: [%d, %d], target: %d\n", tableSubBlockSize, tableSubBlockSize + numOfElements2 - 1, dataTile[threadIdx.x]);
rank1 = binarySearchOdd(dataTile, tableSubBlockSize, tableSubBlockSize + numOfElements2 - 1, dataTile[threadIdx.x]);
rank1 = rank1 - tableSubBlockSize + indexStart2;
printf("index: %d, rank: %d\n", dataOffset + indexStart1 + threadIdx.x, rank1);
}*/

/*if (blockIdx.x == 0 && blockIdx.y == 0) {
printf("thread: %d\n", threadIdx.x);
printf("Search Interval: [%d, %d], target: %d\n", 0, numOfElements1 - 1, dataTile[threadIdx.x + tableSubBlockSize]);
rank2 = binarySearchEven(dataTile, 0, numOfElements1 - 1, dataTile[threadIdx.x + tableSubBlockSize]);
rank2 += indexStart1;
printf("index: %d, rank: %d\n", dataOffset + indexStart2 + threadIdx.x, rank2);
}*/
