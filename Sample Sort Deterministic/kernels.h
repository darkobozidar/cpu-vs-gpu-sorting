#ifndef KERNELS_H
#define KERNELS_H

__global__ void printElemsKernel(el_t *table, uint_t tableLen);
__global__ void printDataKernel(uint_t *table, uint_t tableLen);

__global__ void bitonicSortKernel(el_t *dataTable, data_t *localSamples, uint_t tableLen, order_t sortOrder);

#endif
