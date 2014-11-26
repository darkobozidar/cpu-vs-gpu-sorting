#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"

#include "../Utils/data_types_common.h"


__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder);
__global__ void bitonicMergeGlobalKernel(
    data_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase, order_t sortOrder
);
__global__ void bitonicMergeLocalKernel(
    data_t *table, uint_t tableLen, uint_t step, bool isFirstStepOfPhase, order_t sortOrder
);

#endif
