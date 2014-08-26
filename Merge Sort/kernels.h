#ifndef KERNELS_H
#define KERNELS_H

__global__ void bitonicSortKernel(data_t* array, uint_t arrayLen, uint_t sharedMemSize);

#endif
