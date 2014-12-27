#ifndef KERNELS_H
#define KERNELS_H


template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length);

template <order_t sortOrder>
__global__ void mergeSortKernel(data_t *keys, data_t *values);

template <order_t sortOrder>
__global__ void generateRanksKernel(data_t *dataTable, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize);

template <order_t sortOrder>
__global__ void mergeKernel(
    data_t* keysInput, data_t *valuesInput, data_t* keysOutput, data_t *valuesOutput, uint_t *ranksEven,
    uint_t *ranksOdd, uint_t sortedBlockSize
);

#endif
