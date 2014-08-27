#ifndef KERNELS_H
#define KERNELS_H

__global__ void bitonicSortKernel(data_t* array, uint_t arrayLen, uint_t sharedMemSize);
__global__ void generateSublocksKernel(data_t* table, uint_t tableLen, uint_t sharedMemSize);

#endif
