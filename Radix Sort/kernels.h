#ifndef KERNELS_H
#define KERNELS_H

__global__ void sortBlockKernel(el_t *table, uint_t startBit, bool orderAsc);
__global__ void generateBlocksKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t startBit);

#endif
