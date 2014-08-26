#include <stdio.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "kernels.h"


void deviceMemoryInit(data_t* inputHost, data_t** arrayDevice, uint_t arrayLen) {
	cudaError_t error;

	error = cudaMalloc(arrayDevice, arrayLen * sizeof(*arrayDevice));
	checkCudaError(error);
	error = cudaMemcpy(*arrayDevice, inputHost, arrayLen * sizeof(*arrayDevice), cudaMemcpyHostToDevice);
	checkCudaError(error);
}

void runBitonicSortKernel(data_t* arrayDevice, uint_t arrayLen, uint_t blockSize, uint_t sharedMemSize) {
	cudaError_t error;
	LARGE_INTEGER timerStart;

	dim3 dimGrid((arrayLen - 1) / (2 * blockSize) + 1, 1, 1);
	dim3 dimBlock(blockSize, 1, 1);

	startStopwatch(&timerStart);
	bitonicSortKernel<<<dimGrid, dimBlock, sharedMemSize * sizeof(*arrayDevice)>>>(arrayDevice, arrayLen, sharedMemSize);
	error = cudaDeviceSynchronize();
	checkCudaError(error);
	endStopwatch(timerStart, "Executing Merge Sort Kernel");
}

void merge() {
	// TODO
}

uint_t sortParallel(data_t* inputHost, data_t* outputHost, uint_t arrayLen, bool orderAsc) {
	data_t* arrayDevice;  // Sort in device is done in place
	data_t* samplesDevice;
	cudaError_t error;

	// Every thread compares 2 elements
	uint_t blockSize = 4;  // arrayLen / 2 < getMaxThreadsPerBlock() ? arrayLen / 2 : getMaxThreadsPerBlock();
	uint_t blocksPerMultiprocessor = getMaxThreadsPerMultiProcessor() / blockSize;
	// TODO fix shared memory size from 46KB to 16KB
	uint_t sharedMemSize = 16384 / sizeof(*inputHost) / blocksPerMultiprocessor;

	deviceMemoryInit(inputHost, &arrayDevice, arrayLen);
	runBitonicSortKernel(arrayDevice, arrayLen, blockSize, sharedMemSize);

	error = cudaMemcpy(outputHost, arrayDevice, arrayLen * sizeof(*outputHost), cudaMemcpyDeviceToHost);
	checkCudaError(error);
	
	for (int i = 0; i < arrayLen; i += blockSize) {
		printf("%d ", outputHost[i]);
	}
	printf("\n\n");
	
	return sharedMemSize;
}
