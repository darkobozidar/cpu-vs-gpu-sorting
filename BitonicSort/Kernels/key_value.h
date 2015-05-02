#ifndef KERNELS_KEY_VALUE_BITONIC_SORT_H
#define KERNELS_KEY_VALUE_BITONIC_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "key_value_utils.h"


/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <uint_t threadsBitonicSort, uint_t elemsBitonicSort, order_t sortOrder>
__global__ void bitonicSortKernel(data_t *keys, data_t *values, uint_t tableLen)
{
    normalizedBitonicSort<threadsBitonicSort, elemsBitonicSort, sortOrder>(
        keys, values, keys, values, tableLen
    );
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory size.
*/
template <uint_t threadsMerge, uint_t elemsMerge, order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeGlobalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    uint_t offset, dataBlockLength;
    calcDataBlockLength<threadsMerge, elemsMerge>(offset, dataBlockLength, tableLen);

    bitonicMergeStep<threadsMerge, sortOrder, isFirstStepOfPhase>(
        keys, values, offset / 2, tableLen, dataBlockLength, 1 << (step - 1)
    );
}

/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory size.
*/
template <uint_t threadsMerge, uint_t elemsMerge, order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    bitonicMergeLocal<threadsMerge, elemsMerge, sortOrder, isFirstStepOfPhase>(keys, values, tableLen, step);
}

#endif
