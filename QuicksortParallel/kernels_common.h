#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"


template <uint_t threadsReduction, uint_t elemsThreadReduction>
__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen);

#endif
