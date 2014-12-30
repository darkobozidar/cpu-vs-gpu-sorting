#ifndef KERNELS_H
#define KERNELS_H


__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen);
__global__ void quickSortGlobalKernel(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, data_t *pivotValues,
    d_glob_seq_t *sequences, uint_t *seqIndexes
);
template <order_t sortOrder>
__global__ void quickSortLocalKernel(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, loc_seq_t *sequences
);

#endif
