#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"

#include "../Utils/data_types_common.h"


template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length);

template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen);

template <order_t sortOrder>
__global__ void initIntervalsKernel(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
);
template <order_t sortOrder>
__global__ void generateIntervalsKernel(
    data_t *table, interval_t *input, interval_t *output, uint_t tableLen, uint_t phase, uint_t stepStart,
    uint_t stepEnd
);

template <order_t sortOrder>
__global__ void bitonicMergeKernel(data_t *input, data_t *output, interval_t *intervals, uint_t phase);

#endif
