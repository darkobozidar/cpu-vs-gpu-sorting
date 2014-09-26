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

        uint2 rank = split((el0.key >> shift) & 1, (el1.key >> shift) & 1);

        sortTile[rank.x] = el0;
        sortTile[rank.y] = el1;
        __syncthreads();
    }

    table[index] = sortTile[threadIdx.x];
    table[index + blockDim.x] = sortTile[threadIdx.x + blockDim.x];
}
