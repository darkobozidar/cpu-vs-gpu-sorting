#ifndef DEVICE_PROPS_H
#define DEVICE_PROPS_H

cudaDeviceProp getCudaDeviceProp(int deviceIndex);
cudaDeviceProp getCudaDeviceProp();
int getMaxThreadsPerBlock(int deviceIndex);
int getMaxThreadsPerBlock();
int getMaxThreadsPerMultiProcessor(int deviceIndex);
int getMaxThreadsPerMultiProcessor();
int getMultiProcessorCount(int deviceIndex);
int getMultiProcessorCount();
int getSharedMemoryPerBlock(int deviceIndex);
int getSharedMemoryPerBlock();
int getSharedMemoryPerMultiprocesor(int deviceIndex);
int getSharedMemoryPerMultiprocesor();

void checkCudaError(cudaError_t error);

#endif
