#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, bool orderAsc) {
    if (((int_t)(elem1->key - elem2->key) <= 0) ^ orderAsc) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Sorts input data with bitonic sort and outputs them to output array.
- TODO use quick sort kernel instead of bitonic sort
*/
__device__ void bitonicSortKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc) {
    extern __shared__ el_t sortTile[];
    uint_t elementsPerBlock = blockDim.x * ELEMENTS_PER_THREAD_LOCAL;
    uint_t index = blockIdx.x * ELEMENTS_PER_THREAD_LOCAL * blockDim.x + threadIdx.x;

    // Read data from global to shared memory
    for (uint_t i = 0; i < ELEMENTS_PER_THREAD_LOCAL; i++) {
        sortTile[i * blockDim.x + threadIdx.x] = input[i * blockDim.x + index];
    }
    __syncthreads();

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize < elementsPerBlock; subBlockSize <<= 1) {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {

            // Every thread can sort/exchange 2+ elements
            for (uint_t offsetFactor = 0; offsetFactor < ELEMENTS_PER_THREAD_LOCAL / 2; offsetFactor++) {
                // TODO check if bottom 2 statements can be moved outside this for loop
                uint_t tx = offsetFactor * blockDim.x + threadIdx.x;
                bool direction = orderAsc ^ ((tx & subBlockSize) != 0);

                uint_t start = 2 * tx - (tx & (stride - 1));
                compareExchange(&sortTile[start], &sortTile[start + stride], direction);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t i = 0; i < ELEMENTS_PER_THREAD_LOCAL; i++) {
        output[i * blockDim.x + index] = sortTile[i * blockDim.x + threadIdx.x];
    }
}

__global__ void quickSortLocalKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc) {
    bitonicSortKernel(input, output, tableLen, orderAsc);
}
