#ifndef KERNELS_H
#define KERNELS_H

__global__ void sortBlockKernel(el_t *table, uint_t digit, bool orderAsc);

#endif
