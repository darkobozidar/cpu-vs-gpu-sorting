#ifndef KERNELS_COMMON_BITONIC_SORT_ADAPTIVE_H
#define KERNELS_COMMON_BITONIC_SORT_ADAPTIVE_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"


/*
Adds the padding to table from start index (original table length, which is not power of 2) to the end of the
extended array (which is the next power of 2 of the original table length). Needed because of bitonic sort, for
which table length divisable by 2 is needed.
*/
template <data_t value, data_t threadsPadding, uint_t elemsPadding>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length)
{
    uint_t elemsPerThreadBlock = threadsPadding * elemsPadding;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= length ? elemsPerThreadBlock : length - offset;
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsPadding)
    {
        uint_t index = offset + tx;
        dataTable[index] = value;
        dataBuffer[index] = value;
    }
}

#endif
