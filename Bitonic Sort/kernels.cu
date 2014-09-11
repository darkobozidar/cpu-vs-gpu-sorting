#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"


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
Sorts sub-blocks of input data with bitonic sort.
*/
__global__ void bitonicSortKernel(el_t *table, bool orderAsc) {
    extern __shared__ el_t sortTile[];

    // Every thread loads 2 elements
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sortTile[threadIdx.x] = table[index];
    sortTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];

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
    table[index] = sortTile[threadIdx.x];
    table[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}
