#ifndef KERNELS_H
#define KERNELS_H

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen, bool firstRun);
__global__ void quickSortGlobalKernel(el_t *input, el_t *output, d_glob_seq_t *globalParams, uint_t *seqIndexes,
                                      uint_t tableLen);
__global__ void quickSortLocalKernel(el_t *input, el_t *output, loc_seq_t *localParams, uint_t tableLen,
                                     bool orderAsc);

#endif
