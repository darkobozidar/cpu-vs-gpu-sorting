#ifndef KERNELS_H
#define KERNELS_H

__global__ void bitonicSortKernel(el_t *input, el_t *output, bool orderAsc);
__global__ void generateRanksKernel(el_t* table, uint_t* ranks, uint_t dataLen, uint_t sortedBlockSize);
__global__ void mergeKernel(el_t* input, el_t* output, uint_t *ranks, uint_t tableLen, uint_t ranksLen,
                            uint_t sortedBlockSize, uint_t subBlockSize);

#endif
