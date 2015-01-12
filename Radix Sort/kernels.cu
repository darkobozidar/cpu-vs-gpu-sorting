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

/*
Generates lane mask needed to calculate warp scan of predicates.
*/
__device__ uint_t laneMask()
{
    uint_t mask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    return mask;
}

/*
Performs scan for each warp.
*/
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

/*
Performs scan for provided predicates and returns structure of results for each predicate.
*/
__device__ uint4 intraBlockScan(bool pred0, bool pred1, bool pred2, bool pred3)
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / warpSize;
    uint_t laneIdx = threadIdx.x & (warpSize - 1);
    uint_t warpResult = 0;
    uint4 trueBefore;

    warpResult += binaryWarpScan(pred0);
    warpResult += binaryWarpScan(pred1);
    warpResult += binaryWarpScan(pred2);
    warpResult += binaryWarpScan(pred3);
    __syncthreads();

    if (laneIdx == warpSize - 1)
    {
        scanTile[warpIdx] = warpResult + pred0 + pred1 + pred2 + pred3;
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
    trueBefore.z = trueBefore.y + pred1;
    trueBefore.w = trueBefore.z + pred2;

    return trueBefore;
}

/*
Computes the rank of the current element in shared memory block.
*/
__device__ uint4 split(bool pred0, bool pred1, bool pred2, bool pred3)
{
    __shared__ uint_t falseTotal;
    uint4 trueBefore = intraBlockScan(pred0, pred1, pred2, pred3);
    uint4 rank;

    // Last thread computes the total number of elements, which have 'false' predicate value.
    if (threadIdx.x == blockDim.x - 1)
    {
        falseTotal = ELEMS_PER_THREAD_LOCAL * blockDim.x - (trueBefore.w + pred3);
    }
    __syncthreads();

    // Computes the ranks
    rank.x = pred0 ? trueBefore.x + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x) - trueBefore.x;
    rank.y = pred1 ? trueBefore.y + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 1) - trueBefore.y;
    rank.z = pred2 ? trueBefore.z + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 2) - trueBefore.z;
    rank.w = pred3 ? trueBefore.w + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 3) - trueBefore.w;

    return rank;
}

/*---------------------------------------------------------
------------------------- KERNELS -------------------------
-----------------------------------------------------------*/

/*
Adds the padding to table from start index (original table length) to the end of the extended array (divisable
with number of elements processed by one thread block in local radix sort).
*/
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_PADDING)
    {
        dataTable[offset + tx] = value;
    }
}

template __global__ void addPaddingKernel<MIN_VAL>(data_t *dataTable, uint_t start, uint_t length);
template __global__ void addPaddingKernel<MAX_VAL>(data_t *dataTable, uint_t start, uint_t length);


/*
Sorts blocks in shared memory according to current radix diggit. Sort is done for every separatelly for every
bit of diggit.
- TODO use sort order
*/
template <order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *dataTable, uint_t bitOffset)
{
    extern __shared__ data_t sortTile[];
    uint_t offset = blockIdx.x * ELEMS_PER_THREAD_LOCAL * blockDim.x;
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;

    // Every thread reads it's corresponding elements
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_LOCAL_SORT)
    {
        sortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Every thread processes ELEMS_PER_THREAD_LOCAL elements
    for (uint_t shift = bitOffset; shift < bitOffset + BIT_COUNT_PARALLEL; shift++)
    {
        // Every thread reads it's corresponding elements into registers
        data_t el0 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x];
        data_t el1 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 1];
        data_t el2 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 2];
        data_t el3 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 3];
        __syncthreads();

        // Extracts the current bit (predicate) to calculate ranks
        uint4 rank = split(
            (el0 >> shift) & 1, (el1 >> shift) & 1, (el2 >> shift) & 1, (el3 >> shift) & 1
        );

        sortTile[rank.x] = el0;
        sortTile[rank.y] = el1;
        sortTile[rank.z] = el2;
        sortTile[rank.w] = el3;
        __syncthreads();
    }

    // Every thread stores it's corresponding elements
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_LOCAL_SORT)
    {
        dataTable[offset + tx] = sortTile[tx];
    }
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
    uint_t *sizesTile = tile + RADIX_PARALLEL;
    uint_t *radixTile = tile + 2 * RADIX_PARALLEL;

    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    uint_t offset = blockIdx.x * elemsPerLocalSort;

    // Reads current radixes of elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += THREADS_PER_GEN_BUCKETS)
    {
        radixTile[tx] = (dataTable[offset + tx] >> bitOffset) & RADIX_MASK_PARALLEL;
    }
    __syncthreads();

    // Initializes block sizes and offsets
    if (THREADS_PER_GEN_BUCKETS < RADIX_PARALLEL)
    {
        for (int i = 0; i < RADIX_PARALLEL; i += THREADS_PER_GEN_BUCKETS)
        {
            offsetsTile[threadIdx.x + i] = 0;
            sizesTile[threadIdx.x + i] = 0;
        }
    }
    else if (threadIdx.x < RADIX_PARALLEL)
    {
        offsetsTile[threadIdx.x] = 0;
        sizesTile[threadIdx.x] = 0;
    }
    __syncthreads();

    // Search for bucket offsets (where 2 consecutive elements differ)
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += THREADS_PER_GEN_BUCKETS)
    {
        if (tx > 0 && radixTile[tx - 1] != radixTile[tx])
        {
            offsetsTile[radixTile[tx]] = tx;
        }
    }
    __syncthreads();

    // Generate bucket sizes from previously generated bucket offsets
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += THREADS_PER_GEN_BUCKETS)
    {
        if (tx > 0 && radixTile[tx - 1] != radixTile[tx])
        {
            uint_t radix = radixTile[tx - 1];
            sizesTile[radix] = tx - offsetsTile[radix];
        }
    }
    // Size for last bucket
    if (threadIdx.x == THREADS_PER_GEN_BUCKETS - 1)
    {
        uint_t radix = radixTile[elemsPerLocalSort - 1];
        sizesTile[radix] = elemsPerLocalSort - offsetsTile[radix];
    }
    __syncthreads();

    // Stores block offsets and sizes to global memory
    // Block sizes are NOT written in consecutively way, so that global scan can be performed
    if (THREADS_PER_GEN_BUCKETS < RADIX_PARALLEL)
    {
        for (int i = 0; i < RADIX_PARALLEL; i += THREADS_PER_GEN_BUCKETS)
        {
            bucketOffsets[blockIdx.x * RADIX_PARALLEL + threadIdx.x + i] = offsetsTile[threadIdx.x + i];
            bucketSizes[(threadIdx.x + i) * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x + i];
        }
    }
    else if (threadIdx.x < RADIX_PARALLEL)
    {
        bucketOffsets[blockIdx.x * RADIX_PARALLEL + threadIdx.x] = offsetsTile[threadIdx.x];
        bucketSizes[threadIdx.x * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x];
    }
}

/*
From provided offsets scatters elements to their corresponding buckets (according to radix diggit) from
primary to buffer array.
*/
__global__ void radixSortGlobalKernel(
    data_t *dataInput, data_t *dataOutput, uint_t *offsetsLocal, uint_t *offsetsGlobal, uint_t bitOffset
)
{
    extern __shared__ data_t sortGlobalTile[];
    __shared__ uint_t offsetsLocalTile[RADIX_PARALLEL];
    __shared__ uint_t offsetsGlobalTile[RADIX_PARALLEL];
    uint_t radix, indexOutput;

    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    uint_t offset = blockIdx.x * elemsPerLocalSort;

    // Every thread reads multiple elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += THREADS_PER_GLOBAL_SORT)
    {
        sortGlobalTile[tx] = dataInput[offset + tx];
    }

    // Reads local and global offsets
    if (blockDim.x < RADIX_PARALLEL)
    {
        for (int i = 0; i < RADIX_PARALLEL; i += blockDim.x)
        {
            offsetsLocalTile[threadIdx.x + i] = offsetsLocal[blockIdx.x * RADIX_PARALLEL + threadIdx.x + i];
            offsetsGlobalTile[threadIdx.x + i] = offsetsGlobal[(threadIdx.x + i) * gridDim.x + blockIdx.x];
        }
    }
    else if (threadIdx.x < RADIX_PARALLEL)
    {
        offsetsLocalTile[threadIdx.x] = offsetsLocal[blockIdx.x * RADIX_PARALLEL + threadIdx.x];
        offsetsGlobalTile[threadIdx.x] = offsetsGlobal[threadIdx.x * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    // Every thread stores multiple elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += THREADS_PER_GLOBAL_SORT)
    {
        radix = (sortGlobalTile[tx] >> bitOffset) & RADIX_MASK_PARALLEL;
        indexOutput = offsetsGlobalTile[radix] + tx - offsetsLocalTile[radix];
        dataOutput[indexOutput] = sortGlobalTile[tx];
    }
}
