#ifndef KERNELS_COMMON_RADIX_SORT_H
#define KERNELS_COMMON_RADIX_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "key_only_utils.h"


/*
Generates buckets offsets and sizes for every sorted data block, which was sorted with local radix sort.
*/
template <uint_t threadsGenBuckets, uint_t threadsSortLocal, uint_t elemsSortLocal, uint_t radix>
__global__ void generateBucketsKernel(
    data_t *dataTable, uint_t *bucketOffsets, uint_t *bucketSizes, uint_t bitOffset
)
{
    extern __shared__ uint_t tile[];

    // Tile for saving bucket offsets, bucket sizes and radixes
    uint_t *offsetsTile = tile;
    uint_t *sizesTile = tile + radix;
    uint_t *radixTile = tile + 2 * radix;

    const uint_t elemsPerLocalSort = threadsSortLocal * elemsSortLocal;
    const uint_t offset = blockIdx.x * elemsPerLocalSort;

    // Reads current radixes of elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsGenBuckets)
    {
        radixTile[tx] = (dataTable[offset + tx] >> bitOffset) & (radix - 1);
    }
    __syncthreads();

    // Initializes block sizes and offsets
    if (threadsGenBuckets < radix)
    {
        for (int i = 0; i < radix; i += threadsGenBuckets)
        {
            offsetsTile[threadIdx.x + i] = 0;
            sizesTile[threadIdx.x + i] = 0;
        }
    }
    else if (threadIdx.x < radix)
    {
        offsetsTile[threadIdx.x] = 0;
        sizesTile[threadIdx.x] = 0;
    }
    __syncthreads();

    // Search for bucket offsets (where 2 consecutive elements differ)
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsGenBuckets)
    {
        if (tx > 0 && radixTile[tx - 1] != radixTile[tx])
        {
            offsetsTile[radixTile[tx]] = tx;
        }
    }
    __syncthreads();

    // Generate bucket sizes from previously generated bucket offsets
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsGenBuckets)
    {
        if (tx > 0 && radixTile[tx - 1] != radixTile[tx])
        {
            uint_t radix = radixTile[tx - 1];
            sizesTile[radix] = tx - offsetsTile[radix];
        }
    }
    // Size for last bucket
    if (threadIdx.x == threadsGenBuckets - 1)
    {
        uint_t radix = radixTile[elemsPerLocalSort - 1];
        sizesTile[radix] = elemsPerLocalSort - offsetsTile[radix];
    }
    __syncthreads();

    // Stores block offsets and sizes to global memory
    // Block sizes are NOT written in consecutively way, so that global scan can be performed
    if (threadsGenBuckets < radix)
    {
        for (int i = 0; i < radix; i += threadsGenBuckets)
        {
            bucketOffsets[blockIdx.x * radix + threadIdx.x + i] = offsetsTile[threadIdx.x + i];
            bucketSizes[(threadIdx.x + i) * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x + i];
        }
    }
    else if (threadIdx.x < radix)
    {
        bucketOffsets[blockIdx.x * radix + threadIdx.x] = offsetsTile[threadIdx.x];
        bucketSizes[threadIdx.x * gridDim.x + blockIdx.x] = sizesTile[threadIdx.x];
    }
}

#endif
