#ifndef KERNELS_H
#define KERNELS_H

__global__ void bitonicSortKernel(el_t *input, el_t *output, bool orderAsc);
__global__ void generateRanksKernel(el_t* table, uint_t* ranks, uint_t dataLen, uint_t sortedBlockSize,
                                    uint_t subBlockSize);
__global__ void mergeKernel(data_t* inputDataTable, data_t* outputDataTable, uint_t* rankTable, uint_t tableLen,
                            uint_t rankTableLen, uint_t tableBlockSize, uint_t tableSubBlockSize);

#endif
