#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "data_types.h"

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void bitonicSortKernel(el_t *table, bool orderAsc);
__global__ void multiStepKernel(el_t *table, uint_t phase, uint_t step, uint_t degree, bool orderAsc);
__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc);

#endif