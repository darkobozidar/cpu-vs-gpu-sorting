#ifndef KERNELS_H
#define KERNELS_H


__global__ void  quickSortGlobalKernel(
    data_t *dataInput, data_t *dataBuffer, d_glob_seq_t *sequences, uint_t *seqIndexes
);
template <order_t sortOrder>
__global__ void quickSortLocalKernel(data_t *dataInput, data_t *dataBuffer, loc_seq_t *sequences);

#endif
