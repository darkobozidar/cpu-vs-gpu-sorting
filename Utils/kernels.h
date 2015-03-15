#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "kernels_utils.h"


/*
Adds the padding to table from start index to the end of the extended array.
*/
template <data_t threadsPadding, uint_t elemsPadding, bool fillBuffer, data_t value>
__global__ void addPaddingKernel(data_t *arrayPrimary, data_t *arrayBuffer, uint_t start, uint_t paddingLength)
{
    uint_t offset, dataBlockLength;
    calcDataBlockLength<threadsPadding, elemsPadding>(offset, dataBlockLength, paddingLength);
    offset += start;

    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += threadsPadding)
    {
        uint_t index = offset + tx;
        arrayPrimary[index] = value;

        if (fillBuffer)
        {
            arrayBuffer[index] = value;
        }
    }
}

#endif

