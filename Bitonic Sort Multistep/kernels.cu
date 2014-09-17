#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


/****************************
UTILS
*****************************/

__device__ void compareExchange2(el_t *elem1, el_t *elem2, bool orderAsc) {
    if (((int_t)(elem1->key - elem2->key) <= 0) ^ orderAsc) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

__device__ void compareExchange4(el_t *el1, el_t *el2, el_t *el3, el_t *el4, bool direction) {
    compareExchange2(el1, el2, direction);
    compareExchange2(el3, el4, direction);

    compareExchange2(el1, el3, direction);
    compareExchange2(el2, el4, direction);
}

__device__ void compareExchange8(el_t *el1, el_t *el2, el_t *el3, el_t *el4, el_t *el5, el_t *el6,
                                 el_t *el7, el_t *el8, bool direction) {
    compareExchange2(el1, el2, direction);
    compareExchange2(el3, el4, direction);
    compareExchange2(el5, el6, direction);
    compareExchange2(el7, el8, direction);

    compareExchange4(el1, el5, el3, el7, direction);
    compareExchange4(el2, el6, el4, el8, direction);
}

__device__ void compareExchange16(el_t *el1, el_t *el2, el_t *el3, el_t *el4, el_t *el5, el_t *el6, el_t *el7,
                                  el_t *el8, el_t *el9, el_t *el10, el_t *el11, el_t *el12, el_t *el13,
                                  el_t *el14, el_t *el15, el_t *el16, bool direction) {
    compareExchange2(el1, el2, direction);
    compareExchange2(el3, el4, direction);
    compareExchange2(el5, el6, direction);
    compareExchange2(el7, el8, direction);
    compareExchange2(el9, el10, direction);
    compareExchange2(el11, el12, direction);
    compareExchange2(el13, el14, direction);
    compareExchange2(el15, el16, direction);

    compareExchange8(el1, el9, el3, el11, el5, el13, el7, el15, direction);
    compareExchange8(el2, el10, el4, el12, el6, el14, el8, el16, direction);
}

__device__ void load2(el_t *table, uint_t stride, el_t *el1, el_t *el2) {
    *el1 = table[0];
    *el2 = table[stride];
}

__device__ void store2(el_t *table, uint_t stride, el_t el1, el_t el2) {
    table[0] = el1;
    table[stride] = el2;
}

__device__ void load4(el_t *table, uint_t tableOffset, uint_t stride, el_t *el1, el_t *el2,
                      el_t *el3, el_t *el4) {
    load2(table, stride, el1, el2);
    load2(table + tableOffset, stride, el3, el4);
}

__device__ void store4(el_t *table, uint_t tableOffset, uint_t stride, el_t el1, el_t el2,
                      el_t el3, el_t el4) {
    store2(table, stride, el1, el2);
    store2(table + tableOffset, stride, el3, el4);
}

__device__ void load8(el_t *table, uint_t tableOffset, uint_t stride, el_t *el1, el_t *el2, el_t *el3,
                      el_t *el4, el_t *el5, el_t *el6, el_t *el7, el_t *el8) {
    load4(table, tableOffset, stride, el1, el2, el3, el4);
    load4(table + 2 * tableOffset, tableOffset, stride, el5, el6, el7, el8);
}

__device__ void store8(el_t *table, uint_t tableOffset, uint_t stride, el_t el1, el_t el2, el_t el3,
                       el_t el4, el_t el5, el_t el6, el_t el7, el_t el8) {
    store4(table, tableOffset, stride, el1, el2, el3, el4);
    store4(table + 2 * tableOffset, tableOffset, stride, el5, el6, el7, el8);
}

__device__ void load16(el_t *table, uint_t tableOffset, uint_t stride, el_t *el1, el_t *el2, el_t *el3,
                       el_t *el4, el_t *el5, el_t *el6, el_t *el7, el_t *el8, el_t *el9, el_t *el10,
                       el_t *el11, el_t *el12, el_t *el13, el_t *el14, el_t *el15, el_t *el16) {
    load8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    load8(table + 4 * tableOffset, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16);
}

__device__ void store16(el_t *table, uint_t tableOffset, uint_t stride, el_t el1, el_t el2, el_t el3,
                        el_t el4, el_t el5, el_t el6, el_t el7, el_t el8, el_t el9, el_t el10,
                        el_t el11, el_t el12, el_t el13, el_t el14, el_t el15, el_t el16) {
    store8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    store8(table + 4 * tableOffset, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16);
}

/*
Generates parameters needed for multistep.
> stride - (gap) between two elements beeing compared
> threadsPerSubBlocks - how many threads apper per sub-block in current step
> indexTable - start index, at which thread should start fetching elements
> direction - in which direction should elements be sorted
*/
__device__ void getMultiStepParams(uint_t phase, uint_t step, uint_t degree, uint_t &stride,
                                   uint_t &threadsPerSubBlock, uint_t &indexTable, bool &direction) {
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;

    stride = 1 << (step - 1);
    threadsPerSubBlock = 1 << (step - degree);
    indexTable = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    direction = (indexThread >> (phase - degree)) & 1;
}

/****************************
KERNELS
*****************************/

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
            compareExchange2(&sortTile[start], &sortTile[start + stride], direction);
        }
    }

    __syncthreads();
    table[index] = sortTile[threadIdx.x];
    table[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}

__global__ void multiStep1Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
    uint_t stride, tableOffset, indexTable;
    bool direction;
    el_t el1, el2;

    getMultiStepParams(phase, step, 1, stride, tableOffset, indexTable, direction);
    table += indexTable;

    load2(table, stride, &el1, &el2);
    compareExchange2(&el1, &el2, direction ^ orderAsc);
    store2(table, stride, el1, el2);
}

__global__ void multiStep2Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
    uint_t stride, tableOffset, indexTable;
    bool direction;
    el_t el1, el2, el3, el4;

    getMultiStepParams(phase, step, 2, stride, tableOffset, indexTable, direction);
    table += indexTable;

    load4(table, tableOffset, stride, &el1, &el2, &el3, &el4);
    compareExchange4(&el1, &el2, &el3, &el4, direction ^ orderAsc);
    store4(table, tableOffset, stride, el1, el2, el3, el4);
}

__global__ void multiStep3Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
    uint_t stride, tableOffset, indexTable;
    bool direction;
    el_t el1, el2, el3, el4, el5, el6, el7, el8;

    getMultiStepParams(phase, step, 3, stride, tableOffset, indexTable, direction);
    table += indexTable;

    load8(table, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8);
    compareExchange8(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, direction ^ orderAsc);
    store8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
}

__global__ void multiStep4Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
    uint_t stride, tableOffset, indexTable;
    bool direction;
    el_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16;

    getMultiStepParams(phase, step, 4, stride, tableOffset, indexTable, direction);
    table += indexTable;

    load16(
        table, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
    );
    compareExchange16(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13,
        &el14, &el15, &el16, direction ^ orderAsc
    );
    store16(
        table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7,
        el8, el9, el10, el11, el12, el13, el14, el15, el16
    );
}

/*
Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
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
        compareExchange2(&mergeTile[start], &mergeTile[start + stride], direction);
    }

    __syncthreads();
    table[index] = mergeTile[threadIdx.x];
    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
