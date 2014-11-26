#ifndef DEVICE_PROPS_H
#define DEVICE_PROPS_H

#include "cuda_runtime.h"
#include "data_types_common.h"

cudaDeviceProp getCudaDeviceProp(uint_t deviceIndex);
cudaDeviceProp getCudaDeviceProp();
uint_t getMaxThreadsPerBlock(uint_t deviceIndex);
uint_t getMaxThreadsPerBlock();
uint_t getMaxThreadsPerMultiProcessor(uint_t deviceIndex);
uint_t getMaxThreadsPerMultiProcessor();
uint_t getMultiProcessorCount(uint_t deviceIndex);
uint_t getMultiProcessorCount();
uint_t getSharedMemoryPerBlock(uint_t deviceIndex);
uint_t getSharedMemoryPerBlock();
uint_t getSharedMemoryPerMultiprocesor(uint_t deviceIndex);
uint_t getSharedMemoryPerMultiprocesor();

void checkCudaError(cudaError_t error);

#endif
