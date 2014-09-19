#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "data_types.h"

__global__ void bitonicSortKernel(el_t *table, bool orderAsc);
__global__ void generateIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t step);
__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc);

#endif
