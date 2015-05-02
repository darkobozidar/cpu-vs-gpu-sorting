#ifndef KERNELS_COMMON_UTILS_BITONIC_SORT_MULTISTEP_H
#define KERNELS_COMMON_UTILS_BITONIC_SORT_MULTISTEP_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"


/*
Generates parameters needed for multistep bitonic sort.
> stride - (gap) between two elements being compared
> threadsPerSubBlocks - how many threads appear per sub-block in current step
> indexTable - start index, at which thread should start reading elements
*/
inline __device__ void getMultiStepParams(
    uint_t step, uint_t degree, uint_t &stride, uint_t &threadsPerSubBlock, uint_t &indexTable
)
{
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;

    stride = 1 << (step - 1);
    threadsPerSubBlock = 1 << (step - degree);
    indexTable = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
}

#endif
