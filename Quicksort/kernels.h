#ifndef KERNELS_H
#define KERNELS_H

__global__ void quickSortLocalKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc);

#endif
