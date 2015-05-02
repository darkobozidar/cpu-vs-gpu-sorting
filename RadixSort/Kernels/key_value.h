#ifndef KERNELS_KEY_VALUE_RADIX_SORT_H
#define KERNELS_KEY_VALUE_RADIX_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "key_value_utils.h"


/*
Sorts blocks in shared memory according to current radix digit. Sort is done for every separately for every
bit of digit.
Function template cannot be used for "elements per thread" because it has to be processed by preprocessor.
- TODO implement for sort order DESC
*/
template <uint_t threadsSortLocal, uint_t bitCountRadix, order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *keys, data_t *values, uint_t bitOffset)
{
    extern __shared__ data_t sortLocalTile[];
    const uint_t elemsPerThreadBlock = threadsSortLocal * ELEMS_LOCAL_KV;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    __shared__ uint_t falseTotal;
    uint_t index = 0;

    data_t *keysTile = sortLocalTile;
    data_t *valuesTile = sortLocalTile + elemsPerThreadBlock;

    // Every thread reads it's corresponding elements
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsSortLocal)
    {
        keysTile[tx] = keys[offset + tx];
        valuesTile[tx] = values[offset + tx];
    }
    __syncthreads();

    // Every thread processes ELEMS_LOCAL_KV elements
    for (uint_t shift = bitOffset; shift < bitOffset + bitCountRadix; shift++)
    {
        uint_t predResult = 0;

        // Every thread reads it's corresponding elements into registers
#if (ELEMS_LOCAL_KV >= 1)
        index = ELEMS_LOCAL_KV * threadIdx.x;
        data_t key0 = keysTile[index];
        data_t val0 = valuesTile[index];
        bool pred0 = (key0 >> shift) & 1;
        predResult += pred0;
#endif
#if (ELEMS_LOCAL_KV >= 2)
        index++;
        data_t key1 = keysTile[index];
        data_t val1 = valuesTile[index];
        bool pred1 = (key1 >> shift) & 1;
        predResult += pred1;
#endif
#if (ELEMS_LOCAL_KV >= 3)
        index++;
        data_t key2 = keysTile[index];
        data_t val2 = valuesTile[index];
        bool pred2 = (key2 >> shift) & 1;
        predResult += pred2;
#endif
#if (ELEMS_LOCAL_KV >= 4)
        index++;
        data_t key3 = keysTile[index];
        data_t val3 = valuesTile[index];
        bool pred3 = (key3 >> shift) & 1;
        predResult += pred3;
#endif
#if (ELEMS_LOCAL_KV >= 5)
        index++;
        data_t key4 = keysTile[index];
        data_t val4 = valuesTile[index];
        bool pred4 = (key4 >> shift) & 1;
        predResult += pred4;
#endif
#if (ELEMS_LOCAL_KV >= 6)
        index++;
        data_t key5 = keysTile[index];
        data_t val5 = valuesTile[index];
        bool pred5 = (key5 >> shift) & 1;
        predResult += pred5;
#endif
#if (ELEMS_LOCAL_KV >= 7)
        index++;
        data_t key6 = keysTile[index];
        data_t val6 = valuesTile[index];
        bool pred6 = (key6 >> shift) & 1;
        predResult += pred6;
#endif
#if (ELEMS_LOCAL_KV >= 7)
        index++;
        data_t key7 = keysTile[index];
        data_t val7 = valuesTile[index];
        bool pred7 = (key7 >> shift) & 1;
        predResult += pred7;
#endif
        __syncthreads();

        // According to provided predicates calculates number of elements with true predicate before this thread.
        uint_t trueBefore = intraBlockScanKeyValue<threadsSortLocal>(
#if (ELEMS_LOCAL_KV >= 1)
            pred0
#endif
#if (ELEMS_LOCAL_KV >= 2)
            ,pred1
#endif
#if (ELEMS_LOCAL_KV >= 3)
            ,pred2
#endif
#if (ELEMS_LOCAL_KV >= 4)
            ,pred3
#endif
#if (ELEMS_LOCAL_KV >= 5)
            ,pred4
#endif
#if (ELEMS_LOCAL_KV >= 6)
            ,pred5
#endif
#if (ELEMS_LOCAL_KV >= 7)
            ,pred6
#endif
#if (ELEMS_LOCAL_KV >= 8)
            ,pred7
#endif
        );

        // Calculates number of all elements with false predicate
        if (threadIdx.x == threadsSortLocal - 1)
        {
            falseTotal = elemsPerThreadBlock - (trueBefore + predResult);
        }
        __syncthreads();

        // Every thread stores it's corresponding elements
#if (ELEMS_LOCAL_KV >= 1)
        index = pred0 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x) - trueBefore;
        keysTile[index] = key0;
        valuesTile[index] = val0;
#endif
#if (ELEMS_LOCAL_KV >= 2)
        trueBefore += pred0;
        index = pred1 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 1) - trueBefore;
        keysTile[index] = key1;
        valuesTile[index] = val1;
#endif
#if (ELEMS_LOCAL_KV >= 3)
        trueBefore += pred1;
        index = pred2 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 2) - trueBefore;
        keysTile[index] = key2;
        valuesTile[index] = val2;
#endif
#if (ELEMS_LOCAL_KV >= 4)
        trueBefore += pred2;
        index = pred3 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 3) - trueBefore;
        keysTile[index] = key3;
        valuesTile[index] = val3;
#endif
#if (ELEMS_LOCAL_KV >= 5)
        trueBefore += pred3;
        index = pred4 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 4) - trueBefore;
        keysTile[index] = key4;
        valuesTile[index] = val4;
#endif
#if (ELEMS_LOCAL_KV >= 6)
        trueBefore += pred4;
        index = pred5 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 5) - trueBefore;
        keysTile[index] = key5;
        valuesTile[index] = val5;
#endif
#if (ELEMS_LOCAL_KV >= 7)
        trueBefore += pred5;
        index = pred6 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 6) - trueBefore;
        keysTile[index] = key6;
        valuesTile[index] = val6;
#endif
#if (ELEMS_LOCAL_KV >= 8)
        trueBefore += pred6;
        index = pred7 ? trueBefore + falseTotal : (ELEMS_LOCAL_KV * threadIdx.x + 7) - trueBefore;
        keysTile[index] = key7;
        valuesTile[index] = val7;
#endif
        __syncthreads();
    }

    // Every thread stores it's corresponding elements to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsSortLocal)
    {
        keys[offset + tx] = keysTile[tx];
        values[offset + tx] = valuesTile[tx];
    }
}

/*
From provided offsets scatters elements to their corresponding buckets (according to radix diggit) from
primary to buffer array.
*/
template <uint_t threadsSortGlobal, uint_t threadsSortLocal, uint_t elemsSortLocal, uint_t radixParam>
__global__ void radixSortGlobalKernel(
    data_t *keysInput, data_t *valuesInput, data_t *keysOutput, data_t *valuesOutput, uint_t *offsetsLocal,
    uint_t *offsetsGlobal, uint_t bitOffset
)
{
    extern __shared__ data_t sortGlobalTile[];
    __shared__ uint_t offsetsLocalTile[radixParam];
    __shared__ uint_t offsetsGlobalTile[radixParam];

    const uint_t elemsPerLocalSort = threadsSortLocal * elemsSortLocal;
    const uint_t offset = blockIdx.x * elemsPerLocalSort;

    data_t *keysTile = sortGlobalTile;
    data_t *valuesTile = sortGlobalTile + elemsPerLocalSort;

    // Every thread reads multiple elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsSortGlobal)
    {
        keysTile[tx] = keysInput[offset + tx];
        valuesTile[tx] = valuesInput[offset + tx];
    }

    // Reads local and global offsets
    if (blockDim.x < radixParam)
    {
        for (int i = 0; i < radixParam; i += blockDim.x)
        {
            offsetsLocalTile[threadIdx.x + i] = offsetsLocal[blockIdx.x * radixParam + threadIdx.x + i];
            offsetsGlobalTile[threadIdx.x + i] = offsetsGlobal[(threadIdx.x + i) * gridDim.x + blockIdx.x];
        }
    }
    else if (threadIdx.x < radixParam)
    {
        offsetsLocalTile[threadIdx.x] = offsetsLocal[blockIdx.x * radixParam + threadIdx.x];
        offsetsGlobalTile[threadIdx.x] = offsetsGlobal[threadIdx.x * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    // Every thread stores multiple elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsSortGlobal)
    {
        uint_t radix = (keysTile[tx] >> bitOffset) & (radixParam - 1);
        uint_t indexOutput = offsetsGlobalTile[radix] + tx - offsetsLocalTile[radix];

        keysOutput[indexOutput] = keysTile[tx];
        valuesOutput[indexOutput] = valuesTile[tx];
    }
}

#endif
