#ifndef KERNELS_COMMON_UTILS_SAMPLE_SORT_H
#define KERNELS_COMMON_UTILS_SAMPLE_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"


/*
After sort has been performed, samples are collected and stored to array of local samples.
Because array is always padded to next multiple of "elemsPerThreadBlock", we can always collect
numSamples from data block.
*/
template <uint_t threadsBitonicSort, uint_t elemsBitonicSort, uint_t numSamples>
inline __device__ void collectSamples(data_t *localSamples)
{
    extern __shared__ data_t bitonicSortTile[];

    const uint_t elemsPerThreadBlock = threadsBitonicSort * elemsBitonicSort;
    const uint_t localSamplesDistance = elemsPerThreadBlock / numSamples;
    const uint_t offsetSamples = blockIdx.x * numSamples;

    // Collects the samples on offset "localSampleDistance / 2" in order to collect them as evenly as possible
    for (uint_t tx = threadIdx.x; tx < numSamples; tx += threadsBitonicSort)
    {
        localSamples[offsetSamples + tx] = bitonicSortTile[tx * localSamplesDistance + (localSamplesDistance / 2)];
    }
}

#endif
