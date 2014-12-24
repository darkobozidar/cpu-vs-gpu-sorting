#ifndef KERNELS_H
#define KERNELS_H


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
