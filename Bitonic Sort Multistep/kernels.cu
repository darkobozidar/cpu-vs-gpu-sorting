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
    extern __shared__ el_t tile[];
    uint_t strideGlobal = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable = ((indexThread * threadsPerSubBlock) % strideGlobal) + ((indexThread >> (degree - 1)) % threadsPerSubBlock);
    indexTable += indexThread >> (step - 1) << step;
    uint_t bla = (threadIdx.x >> (degree - 1) << (degree)) + (threadIdx.x % (1 << (degree - 1)));
    bool direction = orderAsc ^ ((indexThread >> (phase - 1)) & 1);

    /*if (phase == 3) {
        printf("%d %d\n", bla, bla + (1 << (degree - 1)));
    }*/

    tile[bla] = table[indexTable];
    tile[bla + (1 << (degree - 1))] = table[indexTable + strideGlobal];

    for (uint_t stride = 1 << (degree - 1); stride > 0; stride >>= 1) {
        __syncthreads();
        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareExchange(&tile[start], &tile[start + stride], direction);
    }

    __syncthreads();
    table[indexTable] = tile[bla];
    table[indexTable + strideGlobal] = tile[bla + (1 << (degree - 1))];
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
