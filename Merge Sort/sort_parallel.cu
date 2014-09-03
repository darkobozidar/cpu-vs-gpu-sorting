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
    return sortedBlockSize;
}

/*
Initializes memory needed for parallel implementation of merge sort.
*/
void memoryInit(data_t *h_inputKeys, data_t *h_inputVals, data_t **d_inputKeys, data_t **d_inputVals,
                data_t **d_outputKeys, data_t **d_outputVals, uint_t arrayLen) {
    cudaError_t error;

    error = cudaMalloc(d_inputKeys, arrayLen * sizeof(*d_inputKeys));
    checkCudaError(error);
    error = cudaMalloc(d_inputVals, arrayLen * sizeof(*d_inputVals));
    checkCudaError(error);
    error = cudaMalloc(d_outputKeys, arrayLen * sizeof(*d_outputKeys));
    checkCudaError(error);
    error = cudaMalloc(d_outputVals, arrayLen * sizeof(*d_outputVals));
    checkCudaError(error);

    error = cudaMemcpy(*d_inputKeys, h_inputKeys, arrayLen * sizeof(*d_inputKeys), cudaMemcpyHostToDevice);
    checkCudaError(error);
    error = cudaMemcpy(*d_inputVals, h_inputVals, arrayLen * sizeof(*d_inputVals), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Sorts data blocks of size sortedBlockSize with bitonic sort.
*/
void runBitonicSortKernel(data_t *d_inputKeys, data_t *d_inputVals, data_t *d_outputKeys, data_t *d_outputVals,
                          uint_t arrayLen, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t sharedMemSize = min(arrayLen, MAX_SHARED_MEM_SIZE);
    dim3 dimGrid((arrayLen - 1) / sharedMemSize + 1, 1, 1);
    dim3 dimBlock(sharedMemSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, 2 * sharedMemSize * sizeof(*d_inputKeys)>>>(
        d_inputKeys, d_inputVals, d_outputKeys, d_outputVals, orderAsc
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing Bitonic sort Kernel");
}

/*
Generates ranks of sub-blocks that need to be merged.
*/
void runGenerateRanksKernel(data_t* data, uint_t* ranks, uint_t dataLen, uint_t sortedBlockSize,
                            uint_t subBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t ranksPerSharedMem = MAX_SHARED_MEM_SIZE / sizeof(sample_el_t);
    uint_t numAllRanks = dataLen / subBlockSize;
    uint_t threadBlockSize = min(ranksPerSharedMem, numAllRanks);

    dim3 dimGrid((numAllRanks - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateRanksKernel<<<dimGrid, dimBlock, threadBlockSize * sizeof(sample_el_t)>>>(
        data, ranks, dataLen, sortedBlockSize, subBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    //endStopwatch(timer, "Executing Generate ranks kernel");
}

/*
Executes merge kernel, which merges all consecutive sorted blocks in data.
*/
void runMergeKernel(data_t* inputData, data_t* outputData, uint_t* ranks, uint_t dataLen,
                    uint_t ranksLen, uint_t sortedBlockSize, uint_t tabSubBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t subBlocksPerMergedBlock = sortedBlockSize / tabSubBlockSize * 2;
    uint_t numMergedBlocks = dataLen / (sortedBlockSize * 2);
    uint_t sharedMemSize = tabSubBlockSize * sizeof(*inputData) * 2;
    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
    dim3 dimBlock(tabSubBlockSize, 1, 1);

    startStopwatch(&timer);
    mergeKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        inputData, outputData, ranks, dataLen, ranksLen, sortedBlockSize, tabSubBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    //endStopwatch(timer, "Executing merge kernel");
}

void sortParallel(data_t *h_inputKeys, data_t *h_inputVals, data_t *h_outputKeys, data_t *h_outputVals,
                  uint_t arrayLen, bool orderAsc) {
    data_t *d_inputKeys, *d_inputVals, *d_outputKeys, *d_outputVals;
    cudaError_t error;

    memoryInit(h_inputKeys, h_inputVals, &d_inputKeys, &d_inputVals, &d_outputKeys, &d_outputVals, arrayLen);

    runBitonicSortKernel(d_inputKeys, d_inputVals, d_outputKeys, d_outputVals, arrayLen, orderAsc);

    //// TODO verify, if ALL (also up) device syncs are necessary
    //for (; sortedBlockSize < dataLen; sortedBlockSize *= 2) {
    //    runGenerateRanksKernel(inputDataDevice, ranksDevice, dataLen, sortedBlockSize, subBlockSize);

    //    runMergeKernel(inputDataDevice, outputDataDevice, ranksDevice, dataLen, ranksLen,
    //                   sortedBlockSize, subBlockSize);

    //    data_t* temp = inputDataDevice;
    //    inputDataDevice = outputDataDevice;
    //    outputDataDevice = temp;
    //}

    error = cudaMemcpy(h_outputKeys, d_outputKeys, arrayLen * sizeof(*h_outputKeys), cudaMemcpyDeviceToHost);
    checkCudaError(error);
    error = cudaMemcpy(h_outputVals, d_outputVals, arrayLen * sizeof(*h_outputVals), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    //return outputDataHost;
}
