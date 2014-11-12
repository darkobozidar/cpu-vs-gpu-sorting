#ifndef KERNELS_H
#define KERNELS_H

__global__ void printTableKernel(el_t *table, uint_t tableLen);

__global__ void bitonicSortKernel(el_t *dataTable, uint_t tableLen, order_t sortOrder);

#endif
