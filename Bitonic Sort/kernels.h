#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "data_types.h"

__global__ void bitonicSortKernel(el_t *table, bool orderAsc);

#endif