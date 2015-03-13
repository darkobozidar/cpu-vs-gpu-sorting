#ifndef KERNELS_KEY_ONLY_BITONIC_PARALLEL_H
#define KERNELS_KEY_ONLY_BITONIC_PARALLEL_H

#include "cuda_runtime.h"

#include "../Utils/data_types_common.h"


template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen);

template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t step);

template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *table, uint_t tableLen, uint_t step);

#endif
