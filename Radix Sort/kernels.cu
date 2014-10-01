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


/*---------------------------------------------------------
-------------------------- UTILS --------------------------
-----------------------------------------------------------*/

__global__ void printTableKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i]);
    }
    printf("\n\n");
}

/*
Performs scan and computes, how many elements have 'true' predicate before current element.
*/
__device__ uint2 scan(bool pred0, bool pred1) {
    extern __shared__ uint_t scanTile[];
    uint2 trueBefore;

    scanTile[threadIdx.x] = pred0;
    scanTile[threadIdx.x + blockDim.x] = pred1;
    __syncthreads();

    // First part of the scan
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;

        if (index < 2 * blockDim.x) {
            scanTile[index] += scanTile[index - stride];
        }
        __syncthreads();
    }

    // Second part of the scan
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

/*
Computes the rank of the current element in shared memory block.
*/
__device__ uint2 split(bool pred0, bool pred1) {
    __shared__ uint_t falseTotal;
    uint2 trueBefore = scan(pred0, pred1);
    uint2 rank;

    // Last thread computes the total number of elements, which have 'false' predicate value.
    if (threadIdx.x == blockDim.x - 1) {
        falseTotal = 2 * blockDim.x - (trueBefore.y + pred1);
    }
    __syncthreads();

    // Computes the rank for the first element (in first half of shared memory)
    if (pred0) {
        rank.x = trueBefore.x + falseTotal;
    } else {
        rank.x = threadIdx.x - trueBefore.x;
    }

    // Computes the rank for the second element (in second half of shared memory)
    if (pred1) {
        rank.y = trueBefore.y + falseTotal;
    } else {
        rank.y = threadIdx.x + blockDim.x - trueBefore.y;
    }

    return rank;
}

/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

/*
Sorts blocks in shared memory according to current radix diggit.
*/
__global__ void radixSortLocalKernel(el_t *table, uint_t bitOffset, bool orderAsc) {
    extern __shared__ el_t sortTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    sortTile[threadIdx.x] = table[index];
    sortTile[threadIdx.x + blockDim.x] = table[index + blockDim.x];

    for (uint_t shift = bitOffset; shift < bitOffset + BIT_COUNT; shift++) {
        el_t el0 = sortTile[threadIdx.x];
        el_t el1 = sortTile[threadIdx.x + blockDim.x];
        __syncthreads();

        // Extracts the current bit (predicate) to calculate ranks
        uint2 rank = split((el0.key >> shift) & 1, (el1.key >> shift) & 1);

        sortTile[rank.x] = el0;
        sortTile[rank.y] = el1;
        __syncthreads();
    }

    table[index] = sortTile[threadIdx.x];
    table[index + blockDim.x] = sortTile[threadIdx.x + blockDim.x];
}

/*
Generates buckets offsets and sizes for every sorted data block, which was sorted with local radix sort.
*/
__global__ void generateBucketsKernel(el_t *table, uint_t *bucketOffsets, uint_t *bucketSizes, uint_t bitOffset) {
    extern __shared__ uint_t tile[];

    // Tile for saving bucket offsets, bucket sizes and radixes
    uint_t *offsetsTile = tile;
    uint_t *sizesTile = tile + RADIX;
    uint_t *radixTile = tile + 2 * RADIX;
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    // Reads current radixes of elements
    radixTile[threadIdx.x] = (table[index].key >> bitOffset) & (RADIX - 1);
    radixTile[threadIdx.x + blockDim.x] = (table[index + blockDim.x].key >> bitOffset) & (RADIX - 1);
    __syncthreads();

    // Initializes block sizes and offsets
    // TODO blockDim.x < 2 * radix
    if (blockDim.x < RADIX) {
        for (int i = 0; i < RADIX; i += blockDim.x) {
            offsetsTile[threadIdx.x + i] = 0;
            sizesTile[threadIdx.x + i] = 0;
        }
    } else if (threadIdx.x < RADIX) {
        offsetsTile[threadIdx.x] = 0;
        sizesTile[threadIdx.x] = 0;
    }
    __syncthreads();

    // Search for bucket offsets (where 2 consecutive elements differ)
    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x]) {
        offsetsTile[radixTile[threadIdx.x]] = threadIdx.x;
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x]) {
        offsetsTile[radixTile[threadIdx.x + blockDim.x]] = threadIdx.x + blockDim.x;
    }
    __syncthreads();

    // Generate bucket sizes from previously generated bucket offsets
    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x]) {
        uint_t radix = radixTile[threadIdx.x - 1];
        sizesTile[radix] = threadIdx.x - offsetsTile[radix];
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x]) {
        uint_t radix = radixTile[threadIdx.x + blockDim.x - 1];
        sizesTile[radix] = threadIdx.x + blockDim.x - offsetsTile[radix];
    }
    // Size for last bucket
    if (threadIdx.x == blockDim.x - 1) {
        uint_t radix = radixTile[2 * blockDim.x - 1];
        sizesTile[radix] = 2 * blockDim.x - offsetsTile[radix];
    }
    __syncthreads();

    // Write block offsets and sizes to global memory
    // Block sizes are not writtec consecutively way, so that scan can be performed on this table
    // TODO blockDim.x < 2 * radix, TODO verify
    if (blockDim.x < RADIX) {
        for (int i = 0; i < RADIX; i += blockDim.x) {
            bucketOffsets[blockIdx.x * RADIX + threadIdx.x + i] = offsetsTile[threadIdx.x + i];
            bucketSizes[(threadIdx.x + i) * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x + i];
        }
    } else if (threadIdx.x < RADIX) {
        bucketOffsets[blockIdx.x * RADIX + threadIdx.x] = offsetsTile[threadIdx.x];
        bucketSizes[threadIdx.x * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x];
    }

    /*if (blockIdx.x == 1 && threadIdx.x == 0) {
        for (int i = 0; i < radix; i++) {
            printf("%2d, ", bucketOffsets[i]);
        }
        printf("\n\n");

        for (int i = 0; i < radix; i++) {
            printf("%2d, ", bucketSizes[i]);
        }
        printf("\n\n");
    }*/
}

__global__ void sortGlobalKernel(el_t *input, el_t *output, uint_t *offsetsLocal, uint_t *offsetsGlobal,
                                 uint_t bitOffset) {
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
        offsetsGlobalTile[threadIdx.x] = offsetsGlobal[threadIdx.x * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    radix = (sortGlobalTile[threadIdx.x].key >> bitOffset) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x - offsetsLocal[radix];
    output[indexOutput] = sortGlobalTile[threadIdx.x];

    radix = (sortGlobalTile[threadIdx.x + blockDim.x].key >> bitOffset) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x - offsetsLocal[radix];
    output[indexOutput] = sortGlobalTile[threadIdx.x + blockDim.x];
}
