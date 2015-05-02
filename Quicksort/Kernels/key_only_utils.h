#ifndef KERNEL_KEY_ONLY_UTILS_QUICKSORT_H
#define KERNEL_KEY_ONLY_UTILS_QUICKSORT_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../BitonicSort/Kernels/key_only_utils.h"
#include "../data_types.h"


/*
Sorts input data with NORMALIZED bitonic sort (all comparisons are made in same direction,
easy to implement for input sequences of arbitrary size) and outputs them to output array.
*/
template <uint_t threadsBitonicSort, order_t sortOrder>
__device__ void normalizedBitonicSort(data_t *input, data_t *output, loc_seq_t localParams)
{
    extern __shared__ data_t bitonicSortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += threadsBitonicSort)
    {
        bitonicSortTile[tx] = input[localParams.start + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < localParams.length; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            if (stride == subBlockSize)
            {
                bitonicMergeStep<threadsBitonicSort, sortOrder, true>(
                    bitonicSortTile, 0, localParams.length, localParams.length, stride
                );
            }
            else
            {
                bitonicMergeStep<threadsBitonicSort, sortOrder, false>(
                    bitonicSortTile, 0, localParams.length, localParams.length, stride
                );
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += threadsBitonicSort)
    {
        output[localParams.start + tx] = bitonicSortTile[tx];
    }
}

#endif
