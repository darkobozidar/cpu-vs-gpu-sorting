#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "data_types.h"
#include "constants.h"


///*---------------------------------------------------------
//-------------------------- UTILS --------------------------
//-----------------------------------------------------------*/

// TODO remove
__global__ void printTableKernel(data_t *dataTable, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++)
    {
        printf("%2d ", dataTable[i]);
    }
    printf("\n\n");
}

__device__ uint_t laneMask()
{
    uint_t mask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ uint_t binaryWarpScan(bool pred)
{
    uint_t mask = laneMask();
    uint_t ballot = __ballot(pred);
    return __popc(ballot & mask);
}

/*
Performs scan and computes, how many elements have 'true' predicate before current element.
*/
__device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val, uint_t stride)
{
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

    scanTile[index] = 0;
    index += stride;
    scanTile[index] = val;

    if (stride > 1)
    {
        scanTile[index] += scanTile[index - 1];
    }
    if (stride > 2)
    {
        scanTile[index] += scanTile[index - 2];
    }
    if (stride > 4)
    {
        scanTile[index] += scanTile[index - 4];
    }
    if (stride > 8)
    {
        scanTile[index] += scanTile[index - 8];
    }
    if (stride > 16)
    {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

__device__ uint2 intraBlockScan(bool pred0, bool pred1)
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / warpSize;
    uint_t laneIdx = threadIdx.x & (warpSize - 1);
    uint_t predSum = pred0 + pred1;
    uint2 trueBefore;

    uint_t warpResult = binaryWarpScan(pred0);
    warpResult += binaryWarpScan(pred1);
    __syncthreads();

    if (laneIdx == warpSize - 1)
    {
        scanTile[warpIdx] = warpResult + predSum;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < blockDim.x / warpSize)
    {
        scanTile[threadIdx.x] = intraWarpScan(scanTile, scanTile[threadIdx.x], blockDim.x / warpSize);
    }
    __syncthreads();

    trueBefore.x = warpResult + scanTile[warpIdx];
    trueBefore.y = trueBefore.x + pred0;

    return trueBefore;
}

/*
Computes the rank of the current element in shared memory block.
*/
__device__ uint2 split(bool pred0, bool pred1)
{
    __shared__ uint_t falseTotal;
    uint2 trueBefore = intraBlockScan(pred0, pred1);
    uint2 rank;

    // Last thread computes the total number of elements, which have 'false' predicate value.
    if (threadIdx.x == blockDim.x - 1)
    {
        falseTotal = 2 * blockDim.x - (trueBefore.y + pred1);
    }
    __syncthreads();

    // Computes the rank for the first element (in first half of shared memory)
    if (pred0)
    {
        rank.x = trueBefore.x + falseTotal;
    }
    else
    {
        rank.x = 2 * threadIdx.x - trueBefore.x;
    }

    // Computes the rank for the second element (in second half of shared memory)
    if (pred1)
    {
        rank.y = trueBefore.y + falseTotal;
    }
    else
    {
        rank.y = (2 * threadIdx.x + 1) - trueBefore.y;
    }

    return rank;
}

/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

/*
Sorts blocks in shared memory according to current radix diggit.
*/
// TODO read process more than two elements
// TODO use sort order
template <order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *dataTable, uint_t bitOffset)
{
    extern __shared__ data_t sortTile[];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    sortTile[threadIdx.x] = dataTable[index];
    sortTile[threadIdx.x + blockDim.x] = dataTable[index + blockDim.x];
    __syncthreads();

    for (uint_t shift = bitOffset; shift < bitOffset + BIT_COUNT; shift++)
    {
        data_t el0 = sortTile[2 * threadIdx.x];
        data_t el1 = sortTile[2 * threadIdx.x + 1];
        __syncthreads();

        // Extracts the current bit (predicate) to calculate ranks
        uint2 rank = split((el0 >> shift) & 1, (el1 >> shift) & 1);

        sortTile[rank.x] = el0;
        sortTile[rank.y] = el1;
        __syncthreads();
    }

    dataTable[index] = sortTile[threadIdx.x];
    dataTable[index + blockDim.x] = sortTile[threadIdx.x + blockDim.x];
}

template __global__ void radixSortLocalKernel<ORDER_ASC>(data_t *dataTable, uint_t bitOffset);
template __global__ void radixSortLocalKernel<ORDER_DESC>(data_t *dataTable, uint_t bitOffset);


/*
Generates buckets offsets and sizes for every sorted data block, which was sorted with local radix sort.
*/
__global__ void generateBucketsKernel(
    data_t *dataTable, uint_t *bucketOffsets, uint_t *bucketSizes, uint_t bitOffset
)
{
    extern __shared__ uint_t tile[];

    // Tile for saving bucket offsets, bucket sizes and radixes
    uint_t *offsetsTile = tile;
    uint_t *sizesTile = tile + RADIX;
    uint_t *radixTile = tile + 2 * RADIX;
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    // Reads current radixes of elements
    radixTile[threadIdx.x] = (dataTable[index] >> bitOffset) & (RADIX - 1);
    radixTile[threadIdx.x + blockDim.x] = (dataTable[index + blockDim.x] >> bitOffset) & (RADIX - 1);
    __syncthreads();

    // Initializes block sizes and offsets
    // TODO blockDim.x < 2 * radix
    if (blockDim.x < RADIX)
    {
        for (int i = 0; i < RADIX; i += blockDim.x)
        {
            offsetsTile[threadIdx.x + i] = 0;
            sizesTile[threadIdx.x + i] = 0;
        }
    }
    else if (threadIdx.x < RADIX)
    {
        offsetsTile[threadIdx.x] = 0;
        sizesTile[threadIdx.x] = 0;
    }
    __syncthreads();

    // Search for bucket offsets (where 2 consecutive elements differ)
    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x])
    {
        offsetsTile[radixTile[threadIdx.x]] = threadIdx.x;
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x])
    {
        offsetsTile[radixTile[threadIdx.x + blockDim.x]] = threadIdx.x + blockDim.x;
    }
    __syncthreads();

    // Generate bucket sizes from previously generated bucket offsets
    if (threadIdx.x > 0 && radixTile[threadIdx.x - 1] != radixTile[threadIdx.x])
    {
        uint_t radix = radixTile[threadIdx.x - 1];
        sizesTile[radix] = threadIdx.x - offsetsTile[radix];
    }
    if (radixTile[threadIdx.x + blockDim.x - 1] != radixTile[threadIdx.x + blockDim.x])
    {
        uint_t radix = radixTile[threadIdx.x + blockDim.x - 1];
        sizesTile[radix] = threadIdx.x + blockDim.x - offsetsTile[radix];
    }
    // Size for last bucket
    if (threadIdx.x == blockDim.x - 1)
    {
        uint_t radix = radixTile[2 * blockDim.x - 1];
        sizesTile[radix] = 2 * blockDim.x - offsetsTile[radix];
    }
    __syncthreads();

    // Write block offsets and sizes to global memory
    // Block sizes are not writtec consecutively way, so that scan can be performed on this table
    // TODO blockDim.x < 2 * radix, TODO verify
    if (blockDim.x < RADIX)
    {
        for (int i = 0; i < RADIX; i += blockDim.x)
        {
            bucketOffsets[blockIdx.x * RADIX + threadIdx.x + i] = offsetsTile[threadIdx.x + i];
            bucketSizes[(threadIdx.x + i) * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x + i];
        }
    }
    else if (threadIdx.x < RADIX)
    {
        bucketOffsets[blockIdx.x * RADIX + threadIdx.x] = offsetsTile[threadIdx.x];
        bucketSizes[threadIdx.x * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x];
    }
}

__global__ void radixSortGlobalKernel(
    data_t *dataInput, data_t *dataOutput, uint_t *offsetsLocal, uint_t *offsetsGlobal, uint_t bitOffset
)
{
    extern __shared__ data_t sortGlobalTile[];
    __shared__ uint_t offsetsLocalTile[RADIX];
    __shared__ uint_t offsetsGlobalTile[RADIX];
    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    uint_t radix, indexOutput;

    sortGlobalTile[threadIdx.x] = dataInput[index];
    sortGlobalTile[threadIdx.x + blockDim.x] = dataInput[index + blockDim.x];

    if (blockDim.x < RADIX)
    {
        for (int i = 0; i < RADIX; i += blockDim.x)
        {
            offsetsLocalTile[threadIdx.x + i] = offsetsLocal[blockIdx.x * RADIX + threadIdx.x + i];
            offsetsGlobalTile[threadIdx.x + i] = offsetsGlobal[(threadIdx.x + i) * gridDim.x + blockIdx.x];
        }
    }
    else if (threadIdx.x < RADIX)
    {
        offsetsLocalTile[threadIdx.x] = offsetsLocal[blockIdx.x * RADIX + threadIdx.x];
        offsetsGlobalTile[threadIdx.x] = offsetsGlobal[threadIdx.x * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    radix = (sortGlobalTile[threadIdx.x] >> bitOffset) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x - offsetsLocalTile[radix];
    dataOutput[indexOutput] = sortGlobalTile[threadIdx.x];

    radix = (sortGlobalTile[threadIdx.x + blockDim.x] >> bitOffset) & (RADIX - 1);
    indexOutput = offsetsGlobalTile[radix] + threadIdx.x + blockDim.x - offsetsLocalTile[radix];
    dataOutput[indexOutput] = sortGlobalTile[threadIdx.x + blockDim.x];
}
