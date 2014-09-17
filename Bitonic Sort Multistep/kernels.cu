#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


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

__device__ void load2(el_t *table, el_t *el1, el_t *el2, uint_t stride) {
    *el1 = table[0];
    *el2 = table[stride];
}

__device__ void store2(el_t *table, el_t el1, el_t el2, uint_t stride) {
    table[0] = el1;
    table[stride] = el2;
}

__device__ void load4(el_t *table, el_t *el1, el_t *el2, el_t *el3, el_t *el4, uint_t stride, uint_t x) {
    load2(table, el1, el2, stride);
    load2(table + x, el3, el4, stride);
}

__device__ void store4(el_t *table, el_t el1, el_t el2, el_t el3, el_t el4, uint_t stride, uint_t x) {
    store2(table, el1, el2, stride);
    store2(table + x, el3, el4, stride);
}

__device__ void load8(el_t *table, el_t *el1, el_t *el2, el_t *el3, el_t *el4, el_t *el5, el_t *el6, el_t *el7,
                      el_t *el8, uint_t stride, uint_t x) {
    load4(table, el1, el2, el3, el4, stride, x);
    load4(table + 2 * x, el5, el6, el7, el8, stride, x);
}

__device__ void store8(el_t *table, el_t el1, el_t el2, el_t el3, el_t el4, el_t el5, el_t el6, el_t el7,
                       el_t el8, uint_t stride, uint_t x) {
    store4(table, el1, el2, el3, el4, stride, x);
    store4(table + 2 * x, el5, el6, el7, el8, stride, x);
}

__device__ void load16(el_t *table, el_t *el1, el_t *el2, el_t *el3, el_t *el4, el_t *el5, el_t *el6, el_t *el7,
                       el_t *el8, el_t *el9, el_t *el10, el_t *el11, el_t *el12, el_t *el13, el_t *el14,
                       el_t *el15, el_t *el16, uint_t stride, uint_t x) {
    load8(table, el1, el2, el3, el4, el5, el6, el7, el8, stride, x);
    load8(table + 4 * x, el9, el10, el11, el12, el13, el14, el15, el16, stride, x);
}

__device__ void store16(el_t *table, el_t el1, el_t el2, el_t el3, el_t el4, el_t el5, el_t el6, el_t el7,
                        el_t el8, el_t el9, el_t el10, el_t el11, el_t el12, el_t el13, el_t el14,
                        el_t el15, el_t el16, uint_t stride, uint_t x) {
    store8(table, el1, el2, el3, el4, el5, el6, el7, el8, stride, x);
    store8(table + 4 * x, el9, el10, el11, el12, el13, el14, el15, el16, stride, x);
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
            compareExchange2(&sortTile[start], &sortTile[start + stride], direction);
        }
    }

    __syncthreads();
    table[index] = sortTile[threadIdx.x];
    table[blockDim.x + index] = sortTile[blockDim.x + threadIdx.x];
}

__global__ void multiStep1Kernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc) {
    uint_t stride = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    bool direction = orderAsc ^ ((indexThread >> (phase - degree)) & 1);
    el_t el1, el2;

    load2(table + indexTable, &el1, &el2, stride);
    compareExchange2(&el1, &el2, direction);
    store2(table + indexTable, el1, el2, stride);
}

__global__ void multiStep2Kernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc) {
    uint_t stride = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable1 = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    bool direction = orderAsc ^ ((indexThread >> (phase - degree)) & 1);
    el_t el1, el2, el3, el4;

    load4(table + indexTable1, &el1, &el2, &el3, &el4, stride, threadsPerSubBlock);
    compareExchange4(&el1, &el2, &el3, &el4, direction);
    store4(table + indexTable1, el1, el2, el3, el4, stride, threadsPerSubBlock);
}

__global__ void multiStep3Kernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc) {
    uint_t stride = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable1 = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    bool direction = orderAsc ^ ((indexThread >> (phase - degree)) & 1);
    el_t el1, el2, el3, el4, el5, el6, el7, el8;

    load8(table + indexTable1, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, stride, threadsPerSubBlock);
    compareExchange8(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, direction);
    store8(table + indexTable1, el1, el2, el3, el4, el5, el6, el7, el8, stride, threadsPerSubBlock);
}

__global__ void multiStep4Kernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc) {
    uint_t stride = 1 << (step - 1);
    uint_t threadsPerSubBlock = 1 << (step - degree);
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t indexTable1 = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
    bool direction = orderAsc ^ ((indexThread >> (phase - degree)) & 1);
    el_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16;

    load16(table + indexTable1, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11,
        &el12, &el13, &el14, &el15, &el16, stride, threadsPerSubBlock);
    compareExchange16(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13,
        &el14, &el15, &el16, direction);
    store16(table + indexTable1, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13,
        el14, el15, el16, stride, threadsPerSubBlock);
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
        compareExchange2(&mergeTile[start], &mergeTile[start + stride], direction);
    }

    __syncthreads();
    table[index] = mergeTile[threadIdx.x];
    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
}
