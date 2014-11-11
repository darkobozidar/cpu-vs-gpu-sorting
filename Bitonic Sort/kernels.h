#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "data_types.h"

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void bitonicSortKernel(el_t *dataTable, uint_t tableLen, order_t sortOrder);
__global__ void bitonicMergeGlobalKernel(el_t *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase,
                                         order_t sortOrder);
__global__ void bitonicMergeLocalKernel(el_t *table, uint_t phase, bool orderAsc);

#endif