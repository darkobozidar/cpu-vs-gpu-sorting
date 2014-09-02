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

    // Only first half of threads have to do the bitonic sort
    if (threadIdx.x < blockDim.x / 2) {
        // TODO test on bigger tables
        for (uint_t stride = subBlocksPerSortedBlock; stride > 0; stride /= 2) {
            __syncthreads();
            uint_t sampleIndex = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (ranksTile[sampleIndex].sample > ranksTile[sampleIndex + stride].sample) {
                sample_el_t temp = ranksTile[sampleIndex];
                ranksTile[sampleIndex] = ranksTile[sampleIndex + stride];
                ranksTile[sampleIndex + stride] = temp;
            } else if (ranksTile[sampleIndex].sample == ranksTile[sampleIndex + stride].sample && ranksTile[sampleIndex].rank > ranksTile[sampleIndex + stride].rank) {
                sample_el_t temp = ranksTile[sampleIndex];
                ranksTile[sampleIndex] = ranksTile[sampleIndex + stride];
                ranksTile[sampleIndex + stride] = temp;
            }
        }
    }

    // Calculate ranks of current and opposite sorted block in global table
    __syncthreads();
    uint_t rank = ranksTile[threadIdx.x].rank;
    uint_t sample = ranksTile[threadIdx.x].sample;
    uint_t rankDataCurrent = (rank * subBlockSize % sortedBlockSize) + 1;
    uint_t rankDataOpposite = binarySearchRank(data, sortedBlockSize, subBlockSize, rank, sample);

    __syncthreads();
    // Check if rank came from odd or even sorted block
    uint_t oddEvenOffset = (rank / subBlocksPerSortedBlock) % 2;
    // Write ranks, which came from EVEN sorted blocks in FIRST half of table of ranks and
    // write ranks, which came from ODD sorted blocks to SECOND half of ranks table
    ranks[threadIdx.x + oddEvenOffset * blockDim.x] = rankDataCurrent;
    ranks[threadIdx.x + (!oddEvenOffset) * blockDim.x] = rankDataOpposite;

    /*printf("%2d: %d %d\n", sample, ranks[threadIdx.x], oddEvenOffset);
    __syncthreads();
    printOnce("\n");
    printf("%2d: %d %d\n", sample, ranks[threadIdx.x + blockDim.x], oddEvenOffset);
    printOnce("\n\n");*/
}

__device__ int binarySearchEven(data_t* dataTile, int indexStart, int indexEnd, uint_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;
        data_t currSample = dataTile[index];

        if (target < dataTile[index]) {
            indexEnd = index - 1;
        }
        else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__device__ int binarySearchOdd(data_t* dataTile, int indexStart, int indexEnd, uint_t target) {
    while (indexStart <= indexEnd) {
        int index = (indexStart + indexEnd) / 2;
        data_t currSample = dataTile[index];

        if (target <= dataTile[index]) {
            indexEnd = index - 1;
        }
        else {
            indexStart = index + 1;
        }
    }

    return indexStart;
}

__global__ void mergeKernel(data_t* inputDataTable, data_t* outputDataTable, uint_t* rankTable, uint_t tableLen,
                            uint_t rankTableLen, uint_t tableBlockSize, uint_t tableSubBlockSize) {
    extern __shared__ data_t dataTile[];
    uint_t indexRank = blockIdx.y * tableSubBlockSize + blockIdx.x;
    uint_t dataOffset = blockIdx.y * 2 * tableBlockSize;
    uint_t indexStart1, indexStart2, indexEnd1, indexEnd2;
    uint_t index1, index2, rank1, rank2;
    uint_t offset1, offset2;
    uint_t numOfElements1, numOfElements2;

    // TODO read in coalasced way
    if (blockIdx.x > 0) {
        indexStart1 = rankTable[indexRank - 1];
        indexStart2 = rankTable[(indexRank - 1) + rankTableLen / 2];
    } else {
        indexStart1 = 0;
        indexStart2 = 0;
    }

    if (blockIdx.x < tableBlockSize / 2 && indexRank < tableBlockSize * 2) {
        indexEnd1 = rankTable[indexRank];
        indexEnd2 = rankTable[indexRank + rankTableLen / 2];
    } else {
        indexEnd1 = tableBlockSize;
        indexEnd2 = tableBlockSize;
    }

    numOfElements1 = indexEnd1 - indexStart1;
    numOfElements2 = indexEnd2 - indexStart2;
    offset1 = dataOffset + indexStart1;
    offset2 = dataOffset + tableBlockSize + indexStart2;

    /*if (blockIdx.x == 6 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("\n(%u, %u), (%u, %u)\n\n", indexStart1, indexEnd1, indexStart2, indexEnd2);
    }*/

    if (threadIdx.x < numOfElements1) {
        index1 = offset1 + threadIdx.x;
        dataTile[threadIdx.x] = inputDataTable[index1];
    }
    if (threadIdx.x < numOfElements2) {
        index2 = offset2 + threadIdx.x;
        dataTile[threadIdx.x + tableSubBlockSize] = inputDataTable[index2];
    }
    __syncthreads();

    if (threadIdx.x < numOfElements1) {
        /*if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("thread: %d\n", threadIdx.x);
            printf("Search Interval: [%d, %d], target: %d\n", tableSubBlockSize, tableSubBlockSize + numOfElements2 - 1, dataTile[threadIdx.x]);
            rank1 = binarySearchOdd(dataTile, tableSubBlockSize, tableSubBlockSize + numOfElements2 - 1, dataTile[threadIdx.x]);
            rank1 = rank1 - tableSubBlockSize + indexStart2;
            printf("index: %d, rank: %d\n", dataOffset + indexStart1 + threadIdx.x, rank1);
        }*/

        rank1 = binarySearchOdd(dataTile, tableSubBlockSize, tableSubBlockSize + numOfElements2 - 1, dataTile[threadIdx.x]);
        rank1 = rank1 - tableSubBlockSize + indexStart2;
        outputDataTable[dataOffset + indexStart1 + threadIdx.x + rank1] = dataTile[threadIdx.x];
    }
    if (threadIdx.x < numOfElements2) {
        /*if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("thread: %d\n", threadIdx.x);
            printf("Search Interval: [%d, %d], target: %d\n", 0, numOfElements1 - 1, dataTile[threadIdx.x + tableSubBlockSize]);
            rank2 = binarySearchEven(dataTile, 0, numOfElements1 - 1, dataTile[threadIdx.x + tableSubBlockSize]);
            rank2 += indexStart1;
            printf("index: %d, rank: %d\n", dataOffset + indexStart2 + threadIdx.x, rank2);
        }*/

        rank2 = binarySearchEven(dataTile, 0, numOfElements1 - 1, dataTile[threadIdx.x + tableSubBlockSize]);
        rank2 += indexStart1;
        outputDataTable[dataOffset + indexStart2 + threadIdx.x + rank2] = dataTile[threadIdx.x + tableSubBlockSize];
    }
}

/*
Old bitonic sorti implementation.
*/
//__global__ void bitonicSortKernel(data_t* data, uint_t dataLen, uint_t sortedBlockSize, bool orderAsc) {
//    extern __shared__ data_t tile[];
//    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
//    uint_t numStages = ceil(log2((double)sortedBlockSize));
//
//    if (index < dataLen) {
//        tile[threadIdx.x] = data[index];
//    }
//    if (index + blockDim.x < dataLen) {
//        tile[threadIdx.x + blockDim.x] = data[index + blockDim.x];
//    }
//
//    for (uint_t stage = 0; stage < numStages; stage++) {
//        for (uint_t pass = 0; pass <= stage; pass++) {
//            __syncthreads();
//
//            uint_t pairDistance = 1 << (stage - pass);
//            uint_t blockWidth = 2 * pairDistance;
//            uint_t direction = ((threadIdx.x >> stage) & 0x1) ^ orderAsc;
//            uint_t leftId = (threadIdx.x & (pairDistance - 1)) + (threadIdx.x >> (stage - pass)) * blockWidth;
//            uint_t rightId = leftId + pairDistance;
//
//            compareExchange(&tile[leftId], &tile[rightId], direction);
//        }
//    }
//
//    __syncthreads();
//
//    if (index < dataLen) {
//        data[index] = tile[threadIdx.x];
//    }
//    if (index + blockDim.x < dataLen) {
//        data[index + blockDim.x] = tile[threadIdx.x + blockDim.x];
//    }
//}
