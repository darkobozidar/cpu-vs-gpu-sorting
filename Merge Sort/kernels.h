#ifndef KERNELS_H
#define KERNELS_H


template <data_t value>
__global__ void addPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t start, uint_t length);

template <order_t sortOrder>
__global__ void mergeSortKernel(data_t *input);

template <order_t sortOrder>
__global__ void generateSamplesKernel(data_t *dataTable, sample_t *samples, uint_t sortedBlockSize);

template <order_t sortOrder>
__global__ void generateRanksKernel(
    data_t* dataTable, sample_t *samples, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);

template <order_t sortOrder>
__global__ void mergeKernel(
    data_t* input, data_t* output, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize
);

#endif
