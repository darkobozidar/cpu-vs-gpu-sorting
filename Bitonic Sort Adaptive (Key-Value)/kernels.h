#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"

#include "../Utils/data_types_common.h"


// Kernel for padding
template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length);

// Bitonic sort kernel
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *keys, data_t *values, uint_t tableLen);

// Kernels for generating intervals
template <order_t sortOrder>
__global__ void initIntervalsKernel(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t stepStart, uint_t stepEnd
);
template <order_t sortOrder>
__global__ void generateIntervalsKernel(
    data_t *table, interval_t *inputIntervals, interval_t *outputIntervals, uint_t tableLen, uint_t phase,
    uint_t stepStart, uint_t stepEnd
);

// Bitonic merge from intervals kernel
template <order_t sortOrder>
__global__ void bitonicMergeKernel(
    data_t *keys, data_t *values, data_t *keysBuffer, data_t *valuesBuffer, interval_t *intervals, uint_t phase
);

#endif
