#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "data_types.h"

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void bitonicSortKernel(el_t *table, bool orderAsc);
__global__ void bitonicMergeGlobalKernel(el_t *table, uint_t phase, uint_t step, bool orderAsc);
__global__ void bitonicMergeLocalKernel(el_t *table, uint_t phase, bool orderAsc);

#endif