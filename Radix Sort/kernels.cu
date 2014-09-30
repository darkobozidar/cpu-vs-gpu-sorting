#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__device__ uint2 scan(bool pred0, bool pred1) {
    extern __shared__ uint_t scanTile[];
    uint2 trueBefore;

    scanTile[threadIdx.x] = pred0;
    scanTile[threadIdx.x + blockDim.x] = pred1;
    __syncthreads();

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;

        if (index < 2 * blockDim.x) {
            scanTile[index] += scanTile[index - stride];
        }
        __syncthreads();
    }

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;

        if (index + stride < 2 * blockDim.x) {
            scanTile[index + stride] += scanTile[index];
        }
        __syncthreads();
    }

    // Converts inclusive scan to exclusive
    trueBefore.x = scanTile[threadIdx.x] - pred0;
    trueBefore.y = scanTile[threadIdx.x + blockDim.x] - pred1;

    return trueBefore;
}

__device__ uint2 split(bool pred0, bool pred1) {
    __shared__ uint_t falseTotal;
    uint2 trueBefore = scan(pred0, pred1);
    uint2 rank;

    if (threadIdx.x == blockDim.x - 1) {
        falseTotal = 2 * blockDim.x - (trueBefore.y + pred1);
    }
    __syncthreads();

    if (pred0) {
        rank.x = trueBefore.x + falseTotal;
    } else {
        rank.x = threadIdx.x - trueBefore.x;
    }

    if (pred1) {
        rank.y = trueBefore.y + falseTotal;
    } else {
        rank.y = threadIdx.x + blockDim.x - trueBefore.y;
    }

    return rank;
}

__global__ void sortBlockKernel(el_t *table, uint_t startBit, bool orderAsc) {
    extern __shared__ el_t sortTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    sortTile[threadIdx.x] = table[index];
    sortTile[threadIdx.x + blockDim.x] = table[index + blockDim.x];

    for (uint_t shift = startBit; shift < startBit + BIT_COUNT; shift++) {
        el_t el0 = sortTile[threadIdx.x];
        el_t el1 = sortTile[threadIdx.x + blockDim.x];
        __syncthreads();

        uint2 rank = split((el0.key >> shift) & 1, (el1.key >> shift) & 1);

        sortTile[rank.x] = el0;
        sortTile[rank.y] = el1;
        __syncthreads();
    }

    table[index] = sortTile[threadIdx.x];
    table[index + blockDim.x] = sortTile[threadIdx.x + blockDim.x];
}

__global__ void generateBlocksKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t startBit) {
    extern __shared__ uint_t offsetsTile[];
    uint_t radix = 1 << BIT_COUNT;

    uint_t *sizesTile = offsetsTile + radix;
    uint_t *radixTile = offsetsTile + 2 * radix;
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    radixTile[threadIdx.x] = (table[index].key >> startBit) & (radix - 1);
    radixTile[threadIdx.x + blockDim.x] = (table[index + blockDim.x].key >> startBit) & (radix - 1);
    __syncthreads();

    // Generate block offsets
    if (blockDim.x < radix) {
        for (int i = 0; i < radix; i += blockDim.x) {
            offsetsTile[threadIdx.x + i] = 0;
            sizesTile[threadIdx.x + i] = 0;
        }
    } else if (threadIdx.x < radix) {
        offsetsTile[threadIdx.x] = 0;
        sizesTile[threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x]) {
        offsetsTile[radixTile[threadIdx.x]] = threadIdx.x;
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x]) {
        offsetsTile[radixTile[threadIdx.x + blockDim.x]] = threadIdx.x + blockDim.x;
    }
    __syncthreads();

    // Generate block sizes
    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x]) {
        uint_t radix = radixTile[threadIdx.x - 1];
        sizesTile[radix] = threadIdx.x - offsetsTile[radix];
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x]) {
        uint_t radix = radixTile[threadIdx.x + blockDim.x - 1];
        sizesTile[radix] = threadIdx.x + blockDim.x - offsetsTile[radix];
    }
    // Size for last block
    if (threadIdx.x == blockDim.x - 1) {
        uint_t radix = radixTile[2 * blockDim.x - 1];
        sizesTile[radix] = 2 * blockDim.x - offsetsTile[radix];
    }
    __syncthreads();

    // Write block offsets and sizes to global memory
    // Block sizes are written in uncoalasced way, so that scan can be performed on this table
    if (blockDim.x < radix) {
        for (int i = 0; i < radix; i += blockDim.x) {
            blockOffsets[blockIdx.x * radix + threadIdx.x + i] = offsetsTile[threadIdx.x + i];
            blockSizes[blockIdx.x * radix + threadIdx.x + i] = sizesTile[threadIdx.x + i];
        }
    } else if (threadIdx.x < radix) {
        blockOffsets[blockIdx.x * radix + threadIdx.x] = offsetsTile[threadIdx.x];
        blockSizes[threadIdx.x * gridDim.x * blockDim.x + blockIdx.x] = sizesTile[threadIdx.x];
    }

    /*if (blockIdx.x == 1 && threadIdx.x == 0) {
        for (int i = 0; i < radix; i++) {
            printf("%2d, ", blockOffsets[i]);
        }
        printf("\n\n");

        for (int i = 0; i < radix; i++) {
            printf("%2d, ", blockSizes[i]);
        }
        printf("\n\n");
    }*/
}

__global__ void sortGlobalKernel(el_t *input, el_t *output, uint_t *offsetsLocal, uint_t *offsetsGlobal,
                                 uint_t startBit) {
    extern __shared__ el_t sortGlobalTile[];
    __shared__ uint_t offsetsLocalTile[RADIX];
    __shared__ uint_t offsetsGlobalTile[RADIX];
    __shared__ uint_t sizesTile[RADIX];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    uint_t radix, indexOutput;

    sortGlobalTile[threadIdx.x] = input[index];
    sortGlobalTile[threadIdx.x + blockDim.x] = input[index + blockDim.x];

    if (threadIdx.x < 16) {
        offsetsLocalTile[threadIdx.x] = offsetsLocal[threadIdx.x * RADIX + threadIdx.x];
        offsetsGlobalTile[threadIdx.x] = offsetsGlobal[threadIdx.x * gridDim.x * blockDim.x + blockIdx.x];
    }
    __syncthreads();

    radix = (sortGlobalTile[threadIdx.x].key >> startBit) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x - offsetsLocal[radix];
    output[indexOutput] = sortGlobalTile[threadIdx.x];

    radix = (sortGlobalTile[threadIdx.x + blockDim.x].key >> startBit) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x - offsetsLocal[radix];
    output[indexOutput] = sortGlobalTile[threadIdx.x];
}
