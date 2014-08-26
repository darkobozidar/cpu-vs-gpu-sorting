#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


cudaDeviceProp getCudaDeviceProp(int deviceIndex) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceIndex);
	return deviceProp;
}

cudaDeviceProp getCudaDeviceProp() {
	return getCudaDeviceProp(0);
}

int getMaxThreadsPerBlock(int deviceIndex) {
	return getCudaDeviceProp(deviceIndex).maxThreadsPerBlock;
}

int getMaxThreadsPerBlock() {
	return getMaxThreadsPerBlock(0);
}

int getMaxThreadsPerMultiProcessor(int deviceIndex) {
	return getCudaDeviceProp(deviceIndex).maxThreadsPerMultiProcessor;
}

int getMaxThreadsPerMultiProcessor() {
	return getMaxThreadsPerMultiProcessor(0);
}

int getMultiProcessorCount(int deviceIndex) {
	return getCudaDeviceProp(deviceIndex).multiProcessorCount;
}

int getMultiProcessorCount() {
	return getMultiProcessorCount(0);
}

int getSharedMemoryPerBlock(int deviceIndex) {
	return getCudaDeviceProp(deviceIndex).sharedMemPerBlock;
}

int getSharedMemoryPerBlock() {
	return getSharedMemoryPerBlock(0);
}

int getSharedMemoryPerMultiprocesor(int deviceIndex) {
	return getCudaDeviceProp(deviceIndex).sharedMemPerMultiprocessor;
}

int getSharedMemoryPerMultiprocesor() {
	return getSharedMemoryPerMultiprocesor(0);
}

void checkCudaError(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("Error in CUDA function.\nError: %s\n", cudaGetErrorString(error));
		getchar();
		exit(EXIT_FAILURE);
	}
}
