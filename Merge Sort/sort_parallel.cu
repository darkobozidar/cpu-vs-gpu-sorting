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


/*
Returns the initial size of sorted sub-blocks.

- If table length is lower than max threads per block (every thread loads 2 elements in initial
  bitonic sort kernel), than fewer threads (and steps) are needed for bitonic sort
- If data type used is big (for example double), than only limited ammount of data can be saved
  into shared memory
*/
uint_t getInitSortedBlockSize(uint_t dataElementSizeof, uint_t dataLen) {
    uint_t elementsPerSharedMem = MAX_SHARED_MEM_SIZE / dataElementSizeof;
    uint_t sortedBlockSize = min(min(dataLen, getMaxThreadsPerBlock() * 2), elementsPerSharedMem);
    return 8;
}

/*
Initializes memory needed for parallel implementation of merge sort.
*/
void memoryInit(data_t* inputDataHost, data_t** outputDataHost, data_t** inputDataDevice,
                data_t** outputDataDevice, uint_t** ranksDevice, uint_t dataLen, uint_t ranksLen) {
    cudaError_t error;

    // Host memory
    error = cudaHostAlloc(outputDataHost, dataLen * sizeof(**outputDataHost), cudaHostAllocDefault);
    checkCudaError(error);

    // Device memory
    error = cudaMalloc(inputDataDevice, dataLen * sizeof(**inputDataDevice));
    checkCudaError(error);
    error = cudaMalloc(outputDataDevice, dataLen * sizeof(**outputDataDevice));
    checkCudaError(error);
    error = cudaMalloc(ranksDevice, ranksLen * sizeof(**ranksDevice));
    checkCudaError(error);

    // Memory copy
    error = cudaMemcpy(*inputDataDevice, inputDataHost, dataLen * sizeof(**inputDataDevice),
                       cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Sorts data blocks of size sortedBlockSize with bitonic sort.
*/
void runBitonicSortKernel(data_t* data, uint_t dataLen, uint_t sortedBlockSize, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    dim3 dimGrid((dataLen - 1) / sortedBlockSize + 1, 1, 1);
    dim3 dimBlock(sortedBlockSize / 2, 1, 1);  // Every thread loads / sorts 2 elements.

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, sortedBlockSize * sizeof(*data)>>>(
        data, dataLen, sortedBlockSize, orderAsc
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing Bitonic sort Kernel");
}

/*
Generates ranks of sub-blocks that need to be merged.
*/
void runGenerateRanksKernel(data_t* tableDevice, uint_t* rankTable, uint_t tableLen,
                            uint_t sortedBlockSize, uint_t subBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t ranksPerSharedMem = MAX_SHARED_MEM_SIZE / sizeof(sample_el_t);
    uint_t numAllRanks = tableLen / subBlockSize;
    uint_t threadBlockSize = min(ranksPerSharedMem, numAllRanks);

    dim3 dimGrid((numAllRanks - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateRanksKernel<<<dimGrid, dimBlock, threadBlockSize * sizeof(sample_el_t)>>>(
        tableDevice, rankTable, tableLen, sortedBlockSize, subBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing Generate ranks kernel");
}

void runMergeKernel(data_t* inputTableDevice, data_t* outputTableDevice, uint_t* rankTable, uint_t tableLen,
                    uint_t rankTableLen, uint_t tabBlockSize, uint_t tabSubBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timerStart;

    uint_t subBlocksPerMergedBlock = tabBlockSize / tabSubBlockSize * 2;
    uint_t numMergedBlocks = tableLen / (tabBlockSize * 2);
    uint_t sharedMemSize = tabSubBlockSize * sizeof(*inputTableDevice) * 2;
    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
    dim3 dimBlock(tabSubBlockSize, 1, 1);

    startStopwatch(&timerStart);
    mergeKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        inputTableDevice, outputTableDevice, rankTable, tableLen, rankTableLen, tabBlockSize, tabSubBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timerStart, "Executing merge kernel");
}

data_t* sortParallel(data_t* inputDataHost, uint_t dataLen, bool orderAsc) {
    data_t* outputDataHost;
    data_t* inputDataDevice;
    data_t* outputDataDevice;
    uint_t* ranksDevice;

    uint_t sortedBlockSize = getInitSortedBlockSize(sizeof(*inputDataDevice), dataLen);
    uint_t subBlockSize = sortedBlockSize / 2;
    uint_t ranksLen = (dataLen / subBlockSize) * 2;
    cudaError_t error;

    memoryInit(inputDataHost, &outputDataHost, &inputDataDevice, &outputDataDevice,
               &ranksDevice, dataLen, ranksLen);
    runBitonicSortKernel(inputDataDevice, dataLen, sortedBlockSize, orderAsc);

    // TODO verify, if ALL (also up) device syncs are necessary
    for (; sortedBlockSize < dataLen; sortedBlockSize *= 2) {
        runGenerateRanksKernel(inputDataDevice, ranksDevice, dataLen, sortedBlockSize, subBlockSize);
        error = cudaDeviceSynchronize();
        checkCudaError(error);

        runMergeKernel(inputDataDevice, outputDataDevice, ranksDevice, dataLen, ranksLen,
                       sortedBlockSize, subBlockSize);
        error = cudaDeviceSynchronize();
        checkCudaError(error);

        data_t* temp = inputDataDevice;
        inputDataDevice = outputDataDevice;
        outputDataDevice = temp;
    }

    error = cudaMemcpy(outputDataHost, inputDataDevice, dataLen * sizeof(*outputDataHost),
                       cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return outputDataHost;
}
