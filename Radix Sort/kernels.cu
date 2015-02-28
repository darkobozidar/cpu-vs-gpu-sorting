#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
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
template <uint_t blockSize>
__device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val)
{
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

    scanTile[index] = 0;
    index += min(blockSize, WARP_SIZE);
    scanTile[index] = val;

    if (blockSize >= 2)
    {
        scanTile[index] += scanTile[index - 1];
    }
    if (blockSize >= 4)
    {
        scanTile[index] += scanTile[index - 2];
    }
    if (blockSize >= 8)
    {
        scanTile[index] += scanTile[index - 4];
    }
    if (blockSize >= 16)
    {
        scanTile[index] += scanTile[index - 8];
    }
    if (blockSize >= 32)
    {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

/*
Performs scan for provided predicates and returns structure of results for each predicate.
*/
template <uint_t blockSize>
__device__ uint_t intraBlockScan(
#if (ELEMS_PER_THREAD_LOCAL >= 1)
    bool pred0
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 2)
    ,bool pred1
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 3)
    ,bool pred2
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 4)
    ,bool pred3
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 5)
    ,bool pred4
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 6)
    ,bool pred5
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
    ,bool pred6
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 8)
    ,bool pred7
#endif
)
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / WARP_SIZE;
    uint_t laneIdx = threadIdx.x & (WARP_SIZE - 1);
    uint_t warpResult = 0;
    uint_t predResult = 0;
    uint4 trueBefore;

#if (ELEMS_PER_THREAD_LOCAL >= 1)
    warpResult += binaryWarpScan(pred0);
    predResult += pred0;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 2)
    warpResult += binaryWarpScan(pred1);
    predResult += pred1;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 3)
    warpResult += binaryWarpScan(pred2);
    predResult += pred2;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 4)
    warpResult += binaryWarpScan(pred3);
    predResult += pred3;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 5)
    warpResult += binaryWarpScan(pred4);
    predResult += pred4;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 6)
    warpResult += binaryWarpScan(pred5);
    predResult += pred5;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
    warpResult += binaryWarpScan(pred6);
    predResult += pred6;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 8)
    warpResult += binaryWarpScan(pred7);
    predResult += pred7;
#endif
    __syncthreads();

    if (laneIdx == WARP_SIZE - 1)
    {
        scanTile[warpIdx] = warpResult + predResult;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < blockDim.x / WARP_SIZE)
    {
        scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx];
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
    const uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    __shared__ uint_t falseTotal;

    // Every thread reads it's corresponding elements
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += THREADS_PER_LOCAL_SORT)
    {
        sortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Every thread processes ELEMS_PER_THREAD_LOCAL elements
    for (uint_t shift = bitOffset; shift < bitOffset + BIT_COUNT_PARALLEL; shift++)
    {
        uint_t predResult = 0;

        // Every thread reads it's corresponding elements into registers
#if (ELEMS_PER_THREAD_LOCAL >= 1)
        data_t el0 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x];
        bool pred0 = (el0 >> shift) & 1;
        predResult += pred0;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 2)
        data_t el1 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 1];
        bool pred1 = (el1 >> shift) & 1;
        predResult += pred1;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 3)
        data_t el2 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 2];
        bool pred2 = (el2 >> shift) & 1;
        predResult += pred2;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 4)
        data_t el3 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 3];
        bool pred3 = (el3 >> shift) & 1;
        predResult += pred3;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 5)
        data_t el4 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 4];
        bool pred4 = (el4 >> shift) & 1;
        predResult += pred4;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 6)
        data_t el5 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 5];
        bool pred5 = (el5 >> shift) & 1;
        predResult += pred5;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
        data_t el6 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 6];
        bool pred6 = (el6 >> shift) & 1;
        predResult += pred6;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
        data_t el7 = sortTile[ELEMS_PER_THREAD_LOCAL * threadIdx.x + 7];
        bool pred7 = (el7 >> shift) & 1;
        predResult += pred7;
#endif
        __syncthreads();

        // According to provided predicates calculates number of elements with true predicate before this thread.
        uint_t trueBefore = intraBlockScan<THREADS_PER_LOCAL_SORT>(
#if (ELEMS_PER_THREAD_LOCAL >= 1)
            pred0
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 2)
            ,pred1
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 3)
            ,pred2
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 4)
            ,pred3
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 5)
            ,pred4
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 6)
            ,pred5
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
            ,pred6
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 8)
            ,pred7
#endif
        );

        // Calculates number of all elements with false predicate
        if (threadIdx.x == THREADS_PER_LOCAL_SORT - 1)
        {
            falseTotal = elemsPerThreadBlock - (trueBefore + predResult);
        }
        __syncthreads();

        // Every thread stores it's corresponding elements
#if (ELEMS_PER_THREAD_LOCAL >= 1)
        sortTile[pred0 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x) - trueBefore] = el0;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 2)
        trueBefore += pred0;
        sortTile[pred1 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 1) - trueBefore] = el1;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 3)
        trueBefore += pred1;
        sortTile[pred2 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 2) - trueBefore] = el2;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 4)
        trueBefore += pred2;
        sortTile[pred3 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 3) - trueBefore] = el3;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 5)
        trueBefore += pred3;
        sortTile[pred4 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 4) - trueBefore] = el4;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 6)
        trueBefore += pred4;
        sortTile[pred5 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 5) - trueBefore] = el5;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 7)
        trueBefore += pred5;
        sortTile[pred6 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 6) - trueBefore] = el6;
#endif
#if (ELEMS_PER_THREAD_LOCAL >= 8)
        trueBefore += pred6;
        sortTile[pred7 ? trueBefore + falseTotal : (ELEMS_PER_THREAD_LOCAL * threadIdx.x + 7) - trueBefore] = el7;
#endif
        __syncthreads();
    }

    // Every thread stores it's corresponding elements to global memory
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

    const uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    const uint_t offset = blockIdx.x * elemsPerLocalSort;

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

    const uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    const uint_t offset = blockIdx.x * elemsPerLocalSort;

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
