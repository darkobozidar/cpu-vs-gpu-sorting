#ifndef KERNELS_H
#define KERNELS_H

__global__ void sortBlockKernel(el_t *table, uint_t startBit, bool orderAsc);

#endif
