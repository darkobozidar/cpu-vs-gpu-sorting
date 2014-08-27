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

void runBitonicSortKernel(data_t* tableDevice, uint_t tableLen) {
	cudaError_t error;
	LARGE_INTEGER timerStart;

	// Every thread compares 2 elements
	uint_t blockSize = 4;  // arrayLen / 2 < getMaxThreadsPerBlock() ? arrayLen / 2 : getMaxThreadsPerBlock();
	uint_t blocksPerMultiprocessor = getMaxThreadsPerMultiProcessor() / blockSize;
	// TODO fix shared memory size from 46KB to 16KB
	uint_t sharedMemSize = 16384 / sizeof(*tableDevice) / blocksPerMultiprocessor;

	dim3 dimGrid((tableLen - 1) / (2 * blockSize) + 1, 1, 1);
	dim3 dimBlock(blockSize, 1, 1);

	startStopwatch(&timerStart);
	bitonicSortKernel<<<dimGrid, dimBlock, sharedMemSize * sizeof(*tableDevice)>>>(tableDevice, tableLen, sharedMemSize);
	error = cudaDeviceSynchronize();
	checkCudaError(error);
	endStopwatch(timerStart, "Executing Merge Sort Kernel");
}

void runGenerateSublocksKernel(data_t* tableDevice, uint_t tableLen) {
	cudaError_t error;
	LARGE_INTEGER timerStart;

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(4, 1, 1);

	startStopwatch(&timerStart);
	generateSublocksKernel<<<dimGrid, dimBlock, 8>>>(tableDevice, tableLen, 8);
	error = cudaDeviceSynchronize();
	checkCudaError(error);
	endStopwatch(timerStart, "Executing Generate Sublocks kernel");
}

void sortParallel(data_t* inputHost, data_t* outputHost, uint_t tableLen, bool orderAsc) {
	data_t* tableDevice;  // Sort in device is done in place
	data_t* samplesDevice;
	cudaError_t error;

	deviceMemoryInit(inputHost, &tableDevice, tableLen);
	runBitonicSortKernel(tableDevice, tableLen);
	runGenerateSublocksKernel(tableDevice, tableLen);

	error = cudaMemcpy(outputHost, tableDevice, tableLen * sizeof(*outputHost), cudaMemcpyDeviceToHost);
	checkCudaError(error);
}
