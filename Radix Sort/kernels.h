#ifndef KERNELS_H
#define KERNELS_H


template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, uint_t start, uint_t length);

template <order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *dataTable, uint_t bitOffset);

__global__ void generateBucketsKernel(
    data_t *dataTable, uint_t *bucketOffsets, uint_t *bucketSizes, uint_t bitOffset
);

__global__ void radixSortGlobalKernel(
    data_t *input, data_t *output, uint_t *offsetsLocal, uint_t *offsetsGlobal, uint_t bitOffset
);

#endif
