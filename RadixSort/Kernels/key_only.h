#ifndef KERNELS_KEY_ONLY_RADIX_SORT_H
#define KERNELS_KEY_ONLY_RADIX_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "key_only_utils.h"


/*
Sorts blocks in shared memory according to current radix digit. Sort is done for every separately for every
bit of digit.
Function template cannot be used for "elements per thread" because it has to be processed by preprocessor.
- TODO implement for sort order DESC
*/
template <uint_t threadsSortLocal, uint_t bitCountRadix, order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *dataTable, uint_t bitOffset)
{
    extern __shared__ data_t sortTile[];
    const uint_t elemsPerThreadBlock = threadsSortLocal * ELEMS_LOCAL_KO;
    const uint_t offset = blockIdx.x * elemsPerThreadBlock;
    __shared__ uint_t falseTotal;

    // Every thread reads it's corresponding elements
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsSortLocal)
    {
        sortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Every thread processes ELEMS_LOCAL_KO elements
    for (uint_t shift = bitOffset; shift < bitOffset + bitCountRadix; shift++)
    {
        uint_t predResult = 0;

        // Every thread reads it's corresponding elements into registers
#if (ELEMS_LOCAL_KO >= 1)
        data_t el0 = sortTile[ELEMS_LOCAL_KO * threadIdx.x];
        bool pred0 = (el0 >> shift) & 1;
        predResult += pred0;
#endif
#if (ELEMS_LOCAL_KO >= 2)
        data_t el1 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 1];
        bool pred1 = (el1 >> shift) & 1;
        predResult += pred1;
#endif
#if (ELEMS_LOCAL_KO >= 3)
        data_t el2 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 2];
        bool pred2 = (el2 >> shift) & 1;
        predResult += pred2;
#endif
#if (ELEMS_LOCAL_KO >= 4)
        data_t el3 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 3];
        bool pred3 = (el3 >> shift) & 1;
        predResult += pred3;
#endif
#if (ELEMS_LOCAL_KO >= 5)
        data_t el4 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 4];
        bool pred4 = (el4 >> shift) & 1;
        predResult += pred4;
#endif
#if (ELEMS_LOCAL_KO >= 6)
        data_t el5 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 5];
        bool pred5 = (el5 >> shift) & 1;
        predResult += pred5;
#endif
#if (ELEMS_LOCAL_KO >= 7)
        data_t el6 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 6];
        bool pred6 = (el6 >> shift) & 1;
        predResult += pred6;
#endif
#if (ELEMS_LOCAL_KO >= 7)
        data_t el7 = sortTile[ELEMS_LOCAL_KO * threadIdx.x + 7];
        bool pred7 = (el7 >> shift) & 1;
        predResult += pred7;
#endif
        __syncthreads();

        // According to provided predicates calculates number of elements with true predicate before this thread.
        uint_t trueBefore = intraBlockScanKeyOnly<threadsSortLocal>(
#if (ELEMS_LOCAL_KO >= 1)
            pred0
#endif
#if (ELEMS_LOCAL_KO >= 2)
            , pred1
#endif
#if (ELEMS_LOCAL_KO >= 3)
            , pred2
#endif
#if (ELEMS_LOCAL_KO >= 4)
            , pred3
#endif
#if (ELEMS_LOCAL_KO >= 5)
            , pred4
#endif
#if (ELEMS_LOCAL_KO >= 6)
            , pred5
#endif
#if (ELEMS_LOCAL_KO >= 7)
            , pred6
#endif
#if (ELEMS_LOCAL_KO >= 8)
            , pred7
#endif
            );

        // Calculates number of all elements with false predicate
        if (threadIdx.x == threadsSortLocal - 1)
        {
            falseTotal = elemsPerThreadBlock - (trueBefore + predResult);
        }
        __syncthreads();

        // Every thread stores it's corresponding elements
#if (ELEMS_LOCAL_KO >= 1)
        sortTile[pred0 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x) - trueBefore] = el0;
#endif
#if (ELEMS_LOCAL_KO >= 2)
        trueBefore += pred0;
        sortTile[pred1 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 1) - trueBefore] = el1;
#endif
#if (ELEMS_LOCAL_KO >= 3)
        trueBefore += pred1;
        sortTile[pred2 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 2) - trueBefore] = el2;
#endif
#if (ELEMS_LOCAL_KO >= 4)
        trueBefore += pred2;
        sortTile[pred3 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 3) - trueBefore] = el3;
#endif
#if (ELEMS_LOCAL_KO >= 5)
        trueBefore += pred3;
        sortTile[pred4 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 4) - trueBefore] = el4;
#endif
#if (ELEMS_LOCAL_KO >= 6)
        trueBefore += pred4;
        sortTile[pred5 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 5) - trueBefore] = el5;
#endif
#if (ELEMS_LOCAL_KO >= 7)
        trueBefore += pred5;
        sortTile[pred6 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 6) - trueBefore] = el6;
#endif
#if (ELEMS_LOCAL_KO >= 8)
        trueBefore += pred6;
        sortTile[pred7 ? trueBefore + falseTotal : (ELEMS_LOCAL_KO * threadIdx.x + 7) - trueBefore] = el7;
#endif
        __syncthreads();
    }

    // Every thread stores it's corresponding elements to global memory
    for (uint_t tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsSortLocal)
    {
        dataTable[offset + tx] = sortTile[tx];
    }
}

/*
From provided offsets scatters elements to their corresponding buckets (according to radix digit) from
primary to buffer array.
*/
template <uint_t threadsSortGlobal, uint_t threadsSortLocal, uint_t elemsSortLocal, uint_t radixParam>
__global__ void radixSortGlobalKernel(
    data_t *dataInput, data_t *dataOutput, uint_t *offsetsLocal, uint_t *offsetsGlobal, uint_t bitOffset
)
{
    extern __shared__ data_t sortGlobalTile[];
    __shared__ uint_t offsetsLocalTile[radixParam];
    __shared__ uint_t offsetsGlobalTile[radixParam];

    const uint_t elemsPerLocalSort = threadsSortLocal * elemsSortLocal;
    const uint_t offset = blockIdx.x * elemsPerLocalSort;

    // Every thread reads multiple elements
    for (uint_t tx = threadIdx.x; tx < elemsPerLocalSort; tx += threadsSortGlobal)
    {
        sortGlobalTile[tx] = dataInput[offset + tx];
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
        uint_t radix = (sortGlobalTile[tx] >> bitOffset) & (radixParam - 1);
        uint_t indexOutput = offsetsGlobalTile[radix] + tx - offsetsLocalTile[radix];
        dataOutput[indexOutput] = sortGlobalTile[tx];
    }
}

#endif
