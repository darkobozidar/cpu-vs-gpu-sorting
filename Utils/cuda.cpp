#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"


cudaDeviceProp getCudaDeviceProp(uint_t deviceIndex)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIndex);
    return deviceProp;
}

cudaDeviceProp getCudaDeviceProp()
{
    return getCudaDeviceProp(0);
}

uint_t getMaxThreadsPerBlock(uint_t deviceIndex)
{
    return getCudaDeviceProp(deviceIndex).maxThreadsPerBlock;
}

uint_t getMaxThreadsPerBlock()
{
    return getMaxThreadsPerBlock(0);
}

uint_t getMaxThreadsPerMultiProcessor(uint_t deviceIndex)
{
    return getCudaDeviceProp(deviceIndex).maxThreadsPerMultiProcessor;
}

uint_t getMaxThreadsPerMultiProcessor()
{
    return getMaxThreadsPerMultiProcessor(0);
}

uint_t getMultiProcessorCount(uint_t deviceIndex)
{
    return getCudaDeviceProp(deviceIndex).multiProcessorCount;
}

uint_t getMultiProcessorCount() {
    return getMultiProcessorCount(0);
}

uint_t getSharedMemoryPerBlock(uint_t deviceIndex)
{
    return getCudaDeviceProp(deviceIndex).sharedMemPerBlock;
}

uint_t getSharedMemoryPerBlock()
{
    return getSharedMemoryPerBlock(0);
}

uint_t getSharedMemoryPerMultiprocesor(uint_t deviceIndex)
{
    return getCudaDeviceProp(deviceIndex).sharedMemPerMultiprocessor;
}

uint_t getSharedMemoryPerMultiprocesor()
{
    return getSharedMemoryPerMultiprocesor(0);
}

/*
Checks if there was any error in cuda memory allocation and prints-out error message.
*/
void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        printf("Error in CUDA function.\nError: %s\n", cudaGetErrorString(error));
        getchar();
        exit(EXIT_FAILURE);
    }
}
