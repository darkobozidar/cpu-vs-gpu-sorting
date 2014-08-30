#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__device__ void compare(void* val1, void* val2) {
    // TODO
}

__device__ void compare(data_t* elem1, data_t* elem2) {
    // TODO
}

__device__ void printfOnce(char* text) {
    if (threadIdx.x == 0) {
        printf(text);
    }
}

__global__ void bitonicSortKernel(data_t* array, uint_t arrayLen, uint_t sharedMemSize) {
    extern __shared__ data_t tile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    uint_t numStages = ceil(log2((double) sharedMemSize));

    if (index < arrayLen) {
        tile[threadIdx.x] = array[index];
    }
    if (index + blockDim.x < arrayLen) {
        tile[threadIdx.x + blockDim.x] = array[index + blockDim.x];
    }

    for (uint_t stage = 0; stage < numStages; stage++) {
        for (uint_t pass = 0; pass <= stage; pass++) {
            __syncthreads();

            uint_t pairDistance = 1 << (stage - pass);
            uint_t blockWidth = 2 * pairDistance;
            uint_t leftId = (threadIdx.x & (pairDistance - 1)) + (threadIdx.x >> (stage - pass)) * blockWidth;
            uint_t rightId = leftId + pairDistance;

            data_t leftElement, rightElement;
            data_t greater, lesser;
            leftElement = tile[leftId];
            rightElement = tile[rightId];

            uint_t sameDirectionBlockWidth = threadIdx.x >> stage;
            uint_t sameDirection = sameDirectionBlockWidth & 0x1;

            uint_t temp = sameDirection ? rightId : temp;
            rightId = sameDirection ? leftId : rightId;
            leftId = sameDirection ? temp : leftId;

            bool compareResult = (leftElement < rightElement);
            greater = compareResult ? rightElement : leftElement;
            lesser = compareResult ? leftElement : rightElement;

            tile[leftId] = lesser;
            tile[rightId] = greater;
        }
    }

    __syncthreads();

    if (index < arrayLen) {
        array[index] = tile[threadIdx.x];
    }
    if (index + blockDim.x < arrayLen) {
        array[index + blockDim.x] = tile[threadIdx.x + blockDim.x];
    }
}

__device__ uint_t calculateSampleIndex(uint_t tableBlockSize, uint_t tableSubBlockSize, bool firstHalf) {
    // Thread index for first or second half of the sub-table
    uint_t threadIdxX = threadIdx.x + (!firstHalf) * blockDim.x;
    uint_t subBlocksPerBlock = tableBlockSize / tableSubBlockSize;
    // Index of a block from which thread will read the sample
    uint_t indexBlock = threadIdxX / subBlocksPerBlock;
    // Offset to block (we devide and multiply with same value, to lose the offset to 
    // sub-block inside last block)
    uint_t index = indexBlock * subBlocksPerBlock;
    // Offset for sub-block index inside block for ODD block
    index += ((indexBlock % 2 == 0) * threadIdxX) % subBlocksPerBlock;
    // Offset for sub-block index inside block for EVEN block (index has to be reversed)
    index += ((indexBlock % 2 == 1) * (subBlocksPerBlock - (threadIdxX + 1))) % subBlocksPerBlock;

    return index;
}

__device__ uint_t binarySearch(data_t* table, sample_el_t* sampleTile, uint_t tableBlockSize, uint_t tableSubBlockSize, bool firstHalf) {
    uint_t threadIdxX = threadIdx.x + (!firstHalf) * blockDim.x;
    uint_t rank = sampleTile[threadIdxX].rank;
    uint_t sample = sampleTile[threadIdxX].sample;
    uint_t subBlocksPerBlock = tableBlockSize / tableSubBlockSize;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerBlock;

    uint_t oppositeBlockOffset = (rank / subBlocksPerMergedBlock) * 2 + !((rank % subBlocksPerMergedBlock) / subBlocksPerBlock);
    uint_t oppositeSubBlockOffset = threadIdxX % subBlocksPerMergedBlock - rank % subBlocksPerBlock - 1;

    // Samples shouldn't be considered
    uint_t indexStart = oppositeBlockOffset * tableBlockSize + oppositeSubBlockOffset * tableSubBlockSize + 1;
    uint_t indexEnd = indexStart + tableSubBlockSize - 2;

    // Has to be explicitly converted to int, because it is unsigned
    if (((int) (indexStart - oppositeBlockOffset * tableBlockSize)) >= 0) {
        while (indexStart <= indexEnd) {
            uint_t index = (indexStart + indexEnd) / 2;
            data_t currSample = table[index];

            if (sample <= table[index]) {
                indexEnd = index - 1;
            }
            else {
                indexStart = index + 1;
            }
        }

        return indexStart - oppositeBlockOffset * tableBlockSize;
    }

    return 0;
}

__global__ void generateSublocksKernel(data_t* table, uint_t* rankTable, uint_t tableLen, uint_t tableBlockSize, uint_t tableSubBlockSize) {
    extern __shared__ sample_el_t sampleTile[];
    uint_t sharedMemIdx;
    data_t value;
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x * tableSubBlockSize;
    uint_t subBlocksPerBlock = tableBlockSize / tableSubBlockSize;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerBlock;

    // Values are read in coalesced way...
    if (index < tableLen) {
        value = table[index];
    }
    // ...and than reversed when added to shared memory
    sharedMemIdx = calculateSampleIndex(tableBlockSize, tableSubBlockSize, true);
    sampleTile[sharedMemIdx].sample = value;
    sampleTile[threadIdx.x].rank = sharedMemIdx;

    if (threadIdx.x < blockDim.x / 2) {
        for (uint_t stride = subBlocksPerBlock; stride > 0; stride /= 2) {
            __syncthreads();
            uint_t sampleIndex = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            // TODO use max/min or conditional operator (or something else)
            if (sampleTile[sampleIndex].sample > sampleTile[sampleIndex + stride].sample) {
                sample_el_t temp = sampleTile[sampleIndex];
                sampleTile[sampleIndex] = sampleTile[sampleIndex + stride];
                sampleTile[sampleIndex + stride] = temp;
            }

            if (sampleTile[sampleIndex].sample == sampleTile[sampleIndex + stride].sample && sampleTile[sampleIndex].rank > sampleTile[sampleIndex + stride].rank) {
                sample_el_t temp = sampleTile[sampleIndex];
                sampleTile[sampleIndex] = sampleTile[sampleIndex + stride];
                sampleTile[sampleIndex + stride] = temp;
            }
        }
    }

    // TODO verify if all __syncthreads are needed
    __syncthreads();
    uint_t rank = (sampleTile[threadIdx.x].rank * tableSubBlockSize % tableBlockSize) + 1;
    uint_t oppositeRank = binarySearch(table, sampleTile, tableBlockSize, tableSubBlockSize, true);

    __syncthreads();
    uint_t oddEvenOffset = (sampleTile[threadIdx.x].rank / subBlocksPerBlock) % 2;
    // TODO fix to write in coalesced way
    // TODO comment odd even
    rankTable[threadIdx.x + oddEvenOffset * blockDim.x] = rank;
    rankTable[threadIdx.x + (!oddEvenOffset) * blockDim.x] = oppositeRank;

    /*printf("%2d: %d %d\n", sampleTile[threadIdx.x].sample, rankTable[threadIdx.x], oddEvenOffset);
    __syncthreads();
    printfOnce("\n");
    printf("%2d: %d %d\n", sampleTile[threadIdx.x].sample, rankTable[threadIdx.x + blockDim.x], oddEvenOffset);
    printfOnce("\n\n");*/
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

   /* if (blockIdx.x == 6 && blockIdx.y == 0 && threadIdx.x == 0) {
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
