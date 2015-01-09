#ifndef KERNELS_H
#define KERNELS_H

//__global__ void printTableKernel(uint_t *table, uint_t tableLen);
//
template <order_t sortOrder>
__global__ void radixSortLocalKernel(data_t *dataTable, uint_t bitOffset);
//__global__ void generateBucketsKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t startBit);
//__global__ void radixSortGlobalKernel(el_t *input, el_t *output, uint_t *offsetsLocal, uint_t *offsetsGlobal,
//                                      uint_t startBit);

#endif
