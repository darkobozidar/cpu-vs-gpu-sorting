#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__global__ void printTableKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i].key);
    }
    printf("\n");
}

/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, order_t sortOrder) {
    if (((int_t)(elem1->key - elem2->key) > 0) ^ sortOrder) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
__global__ void bitonicSortKernel(el_t *dataTable, uint_t tableLen, order_t sortOrder) {
    extern __shared__ el_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1) {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT) {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other
                // STEPS. Also, in first step of every phase, offset sizes are generated in ASCENDING order
                // (normalized bitnic sort requires DESCENDING order). Because of that, we can break the loop if
                // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
                // than thread indexes inside every sub-block have to be reversed.
                if (stride == subBlockSize) {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength) {
                    break;
                }

                compareExchange(&bitonicSortTile[index], &bitonicSortTile[index + offset], sortOrder);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory.
*/
__global__ void bitonicMergeGlobalKernel(el_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase,
                                         order_t sortOrder) {
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    uint_t stride = 1 << (step - 1);
    // Every theoretical thread (threads are emulated with ELEMS_PER_THREAD_GLOBAL_MERGE) sorts 2 elements
    uint_t threadOffset = blockIdx.x * (elemsPerThreadBlock >> 1);

    for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_GLOBAL_MERGE) {
        uint_t indexThread = threadOffset + tx;
        uint_t offset = stride;

        // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
        if (firstStepOfPhase) {
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));

        if (index + offset >= tableLen) {
            break;
        }

        el_t el1 = dataTable[index];
        el_t el2 = dataTable[index + offset];

        compareExchange(&el1, &el2, sortOrder);

        dataTable[index] = el1;
        dataTable[index + offset] = el2;
    }
}

/*
Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
__global__ void bitonicMergeLocalKernel(el_t *table, uint_t phase, bool orderAsc) {
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
        compareExchange(&mergeTile[start], &mergeTile[start + stride], (order_t)!direction);
    }

    __syncthreads();
    table[index] = mergeTile[threadIdx.x];
    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
