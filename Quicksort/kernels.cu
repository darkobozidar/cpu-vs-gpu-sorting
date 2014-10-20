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
Sorts input data with NORMALIZED bitonic sort (all comparisons are made in same direction,
easy to implement for input sequences of arbitrary size) and outputs them to output array.

- TODO use quick sort kernel instead of bitonic sort
*/
__device__ void bitonicSortKernel(el_t *input, el_t *output, uint_t start, uint_t length, bool orderAsc) {
    extern __shared__ el_t sortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < length; tx += blockDim.x) {
        sortTile[tx] = input[start + tx];
    }
    __syncthreads();

    // Bitonic sort
    for (uint_t subBlockSize = 1; subBlockSize < length; subBlockSize <<= 1) {
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            // Every thread can sort/exchange 2 or more elements (at least 2 and only power of 2)
            for (uint_t tx = threadIdx.x; tx < (blockDim.x * ELEMENTS_PER_THREAD_LOCAL >> 1); tx += blockDim.x) {
                // In normalized bitonic sort, first step of every phase uses different stride
                // than all other steps.
                uint_t offset = stride == subBlockSize ? ((stride - (tx & (stride - 1))) << 1) - 1 : stride;
                uint_t index = (tx << 1) - (tx & (stride - 1));
                if (index + offset >= length) {
                    break;
                }

                compareExchange(&sortTile[index], &sortTile[index + offset], orderAsc);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < length; tx += blockDim.x) {
        output[start + tx] = sortTile[tx];
    }
}

__global__ void quickSortLocalKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc) {
    uint_t start = 3;
    uint_t length = 1;
    bitonicSortKernel(input, output, start, length, orderAsc);
}
