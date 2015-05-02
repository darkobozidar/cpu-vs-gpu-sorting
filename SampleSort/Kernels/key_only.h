#ifndef KERNELS_KEY_ONLY_SAMPLE_SORT_H
#define KERNELS_KEY_ONLY_SAMPLE_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../BitonicSort/Kernels/key_only_utils.h"
#include "common_utils.h"


/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort, collects samples and stores them to array for
local samples.
*/
template <uint_t threadsBitonicSort, uint_t elemsBitonicSort, uint_t numSamples, order_t sortOrder>
__global__ void bitonicSortCollectSamplesKernel(data_t *dataTable, data_t *localSamples, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    normalizedBitonicSort<threadsBitonicSort, elemsBitonicSort, sortOrder>(dataTable, dataTable, tableLen);
    collectSamples<threadsBitonicSort, elemsBitonicSort, numSamples>(localSamples);
}

#endif
