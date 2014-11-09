#ifndef KERNELS_H
#define KERNELS_H

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void minMaxReductionKernel(el_t *input, data_t *output, uint_t tableLen);
__global__ void quickSortGlobalKernel(el_t *dataInput, el_t *dataBuffer, d_glob_seq_t *sequences, uint_t *seqIndexes);
__global__ void quickSortLocalKernel(el_t *dataInput, el_t *dataBuffer, loc_seq_t *sequences, bool orderAsc);

#endif
