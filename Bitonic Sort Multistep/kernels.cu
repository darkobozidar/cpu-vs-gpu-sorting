#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void printTableKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i]);
    }
    printf("\n\n");
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
__global__ void bitonicSortKernel(el_t *table, bool orderAsc) {
    extern __shared__ el_t sortTile[];
    // If shared memory size is lower than table length, than every block has to be ordered
    // in opposite direction -> bitonic sequence.
    bool blockDirection = orderAsc ^ (blockIdx.x & 1);

    // Every thread loads 2 elements
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sortTile[threadIdx.x] = table[index];
    sortTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize <= blockDim.x; subBlockSize <<= 1) {
        bool direction = blockDirection ^ ((threadIdx.x & subBlockSize) != 0);

        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            __syncthreads();
            uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareExchange(&sortTile[start], &sortTile[start + stride], direction);
        }
    }

    __syncthreads();
    table[index] = sortTile[threadIdx.x];
    table[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}

__global__ void multiStepKernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc) {
    el_t tile[1 << MAX_MULTI_STEP];  // TODO write separated kernel for this
    uint_t tileHalfSize = 1 << (degree - 1);
    uint_t strideGlobal = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    bool direction = orderAsc ^ ((indexThread >> (phase - degree)) & 1);

    for (uint_t i = 0; i < tileHalfSize; i++) {
        uint_t start = indexTable + i * threadsPerSubBlock;

        /*if (phase == 5 && step == 3) {
        printf("%2d %2d %2d %2d\n", threadIdx.x, start, end, direction);
        }*/

        tile[i] = table[start];
        tile[i + tileHalfSize] = table[start + strideGlobal];
    }

    /*printf("%2d %2d %2d %2d\n", tile[0].key, tile[1].key, tile[2].key, tile[3].key);*/

    // Syncthreads is not needed, because every thread proceses an separated subsection of partition
    for (uint_t strideLocal = tileHalfSize; strideLocal > 0; strideLocal >>= 1) {
        for (uint_t i = 0; i < tileHalfSize; i++) {
            uint_t start = 2 * i - (i & (strideLocal - 1));

            /*printf("%2d %2d %2d %2d\n", threadIdx.x, tile[start].key, tile[end].key, direction);*/
            compareExchange(&tile[start], &tile[start + strideLocal], direction);
            /*printf("%2d %2d %2d %2d\n", threadIdx.x, tile[start].key, tile[end].key, direction);*/
        }
    }

    /*__syncthreads();
    if (threadIdx.x == 0) {
        printf("\n\n");
    }
    printf("%2d %2d %2d %2d\n", tile[0].key, tile[1].key, tile[2].key, tile[3].key);*/

    /*if (threadIdx.x == 0) {
        printf("%2d %2d %2d %2d\n", tile[0].key, tile[1].key, tile[2].key, tile[3].key);
    }*/

    for (int i = 0; i < tileHalfSize; i++) {
        uint_t start = indexTable + i * threadsPerSubBlock;
        table[start] = tile[i];
        table[start + strideGlobal] = tile[i + tileHalfSize];
    }
}

/*
Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc) {
    extern __shared__ el_t mergeTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    // Elements inside same sub-block have to be ordered in same direction
    bool direction = orderAsc ^ ((index >> phase) & 1);

    // Every thread loads 2 elements
    mergeTile[threadIdx.x] = table[index];
    mergeTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];

    // Bitonic merge
    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&mergeTile[start], &mergeTile[start + stride], direction);
    }

    __syncthreads();
    table[index] = mergeTile[threadIdx.x];
    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
