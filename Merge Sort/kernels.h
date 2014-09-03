#ifndef KERNELS_H
#define KERNELS_H

__global__ void bitonicSortKernel(el_t *input, el_t *output, bool orderAsc);
__global__ void generateRanksKernel(data_t* table, uint_t* rankTable, uint_t tableLen,
                                    uint_t tabBlockSize, uint_t tabSubBlockSize);
__global__ void mergeKernel(data_t* inputDataTable, data_t* outputDataTable, uint_t* rankTable, uint_t tableLen,
                            uint_t rankTableLen, uint_t tableBlockSize, uint_t tableSubBlockSize);

#endif
