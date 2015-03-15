#ifndef KERNELS_KEY_VALUE_BITONIC_SORT_H
#define KERNELS_KEY_VALUE_BITONIC_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "kernels_key_value_utils.h"


/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder, uint_t threadsBitonicSort, uint_t elemsBitonicSort>
__global__ void bitonicSortKernel(data_t *keys, data_t *values, uint_t tableLen)
{
    normalizedBitonicSort<sortOrder, threadsBitonicSort, elemsBitonicSort>(
        keys, values, keys, values, tableLen
    );
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase, uint_t threadsMerge, uint_t elemsMerge>
__global__ void bitonicMergeGlobalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    uint_t pairsPerThreadBlock = (threadsMerge * elemsMerge) >> 1;
    uint_t offset = blockIdx.x * pairsPerThreadBlock;

    bitonicMergeStep<sortOrder, threadsMerge, elemsMerge, isFirstStepOfPhase>(
        keys, values, offset, tableLen, 1 << (step - 1)
    );
}

/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase, uint_t threadsMerge, uint_t elemsMerge>
__global__ void bitonicMergeLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    bitonicMergeLocal<sortOrder, isFirstStepOfPhase, threadsMerge, elemsMerge>(keys, values, tableLen, step);
}

#endif
