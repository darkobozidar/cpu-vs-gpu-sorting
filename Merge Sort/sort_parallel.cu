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


void deviceMemoryInit(data_t* inputHost, data_t** tableDevice, uint_t** rankTable, uint_t tableLen,
                      uint_t rankTableLen) {
    cudaError_t error;

    error = cudaMalloc(tableDevice, tableLen * sizeof(*tableDevice));
    checkCudaError(error);
    error = cudaMalloc(rankTable, rankTableLen * sizeof(*rankTable));
    checkCudaError(error);

    error = cudaMemcpy(*tableDevice, inputHost, tableLen * sizeof(*tableDevice), cudaMemcpyHostToDevice);
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

void runGenerateSublocksKernel(data_t* tableDevice, uint_t* rankTable, uint_t tableLen,
                               uint_t tabBlockSize, uint_t tabSubBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timerStart;

    // * 2 for table of ranks, which has the same size as table of samples
    uint_t sharedMemSize = tableLen / tabSubBlockSize * sizeof(sample_el_t);
    uint_t blockSize = tableLen / tabSubBlockSize;
    dim3 dimGrid((tableLen - 1) / (2 * blockSize * tabSubBlockSize) + 1, 1, 1);
    dim3 dimBlock(blockSize, 1, 1);

    startStopwatch(&timerStart);
    generateSublocksKernel<<<dimGrid, dimBlock, sharedMemSize>>>(tableDevice, rankTable, tableLen, tabBlockSize, tabSubBlockSize);
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timerStart, "Executing Generate Sublocks kernel");
}

void runMergeKernel(data_t* tableDevice, uint_t tableLen, uint_t tabBlockSize, uint_t tabSubBlockSize) {

}

void sortParallel(data_t* inputHost, data_t* outputHost, uint_t tableLen, bool orderAsc) {
    data_t* tableDevice;  // Sort in device is done in place
    uint_t* rankTableDevice;
    uint_t tabBlockSize = 8;
    uint_t tabSubBlockSize = 4;  // TODO could be constant
    uint_t rankTableLen = tableLen / tabSubBlockSize * 2;
    cudaError_t error;

    deviceMemoryInit(inputHost, &tableDevice, &rankTableDevice, tableLen, rankTableLen);
    runBitonicSortKernel(tableDevice, tableLen);
    runGenerateSublocksKernel(tableDevice, rankTableDevice, tableLen, tabBlockSize, tabSubBlockSize);

    error = cudaMemcpy(outputHost, tableDevice, tableLen * sizeof(*outputHost), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
