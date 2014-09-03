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
Initializes memory needed for parallel implementation of merge sort.
*/
void memoryInit(el_t *h_input, el_t **d_input, el_t **d_output, el_t **d_buffer, uint_t **d_ranks,
                uint_t tableLen, uint_t ranksLen) {
    cudaError_t error;

    error = cudaMalloc(d_input, tableLen * sizeof(**d_input));
    checkCudaError(error);
    error = cudaMalloc(d_output, tableLen * sizeof(**d_output));
    checkCudaError(error);
    error = cudaMalloc(d_buffer, tableLen * sizeof(**d_buffer));
    checkCudaError(error);
    error = cudaMalloc(d_ranks, ranksLen * sizeof(**d_ranks));
    checkCudaError(error);

    error = cudaMemcpy(*d_input, h_input, tableLen * sizeof(**d_input), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Sorts data blocks of size sortedBlockSize with bitonic sort.
*/
void runBitonicSortKernel(el_t *input, el_t *output, uint_t tableLen, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t sharedMemSize = min(tableLen, MAX_SHARED_MEM_SIZE);
    dim3 dimGrid((tableLen - 1) / sharedMemSize + 1, 1, 1);
    dim3 dimBlock(sharedMemSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, sharedMemSize * sizeof(*input)>>>(
        input, output, orderAsc
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    //endStopwatch(timer, "Executing Bitonic sort Kernel");
}

/*
Generates ranks of sub-blocks that need to be merged.
*/
void runGenerateRanksKernel(el_t *table, uint_t *ranks, uint_t tableLen, uint_t sortedBlockSize,
                            uint_t subBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t numAllRanks = tableLen / subBlockSize;
    uint_t threadBlockSize = min(numAllRanks, MAX_SHARED_MEM_SIZE);
    dim3 dimGrid((numAllRanks - 1) / threadBlockSize + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateRanksKernel<<<dimGrid, dimBlock, threadBlockSize * sizeof(rank_el_t)>>>(
        table, ranks, tableLen, sortedBlockSize, subBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    //endStopwatch(timer, "Executing Generate ranks kernel");
}

/*
Executes merge kernel, which merges all consecutive sorted blocks in data.
*/
void runMergeKernel(el_t *input, el_t *output, uint_t *ranks, uint_t tableLen,
                    uint_t ranksLen, uint_t sortedBlockSize, uint_t tabSubBlockSize) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t subBlocksPerMergedBlock = sortedBlockSize / tabSubBlockSize * 2;
    uint_t numMergedBlocks = tableLen / (sortedBlockSize * 2);
    dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
    dim3 dimBlock(tabSubBlockSize, 1, 1);

    startStopwatch(&timer);
    mergeKernel<<<dimGrid, dimBlock, MAX_SHARED_MEM_SIZE * sizeof(*input) * 2>>>(
        input, output, ranks, tableLen, ranksLen, sortedBlockSize, tabSubBlockSize
    );
    error = cudaDeviceSynchronize();
    checkCudaError(error);
    //endStopwatch(timer, "Executing merge kernel");
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_input, *d_output, *d_buffer;
    uint_t* d_ranks;
    uint_t subBlockSize = min(tableLen, MAX_SHARED_MEM_SIZE);
    uint_t ranksLen = tableLen / subBlockSize * 2;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_input, &d_input, &d_output, &d_buffer, &d_ranks, tableLen, ranksLen);

    startStopwatch(&timer);
    runBitonicSortKernel(d_input, d_output, tableLen, orderAsc);

    for (uint_t sortedBlockSize = MAX_SHARED_MEM_SIZE; sortedBlockSize < tableLen; sortedBlockSize *= 2) {
        el_t* temp = d_output;
        d_output = d_buffer;
        d_buffer = temp;

        runGenerateRanksKernel(d_buffer, d_ranks, tableLen, sortedBlockSize, subBlockSize);
        runMergeKernel(d_buffer, d_output, d_ranks, tableLen, ranksLen, sortedBlockSize, subBlockSize);
    }
    endStopwatch(timer, "Executing parallel merge sort.");

    error = cudaMemcpy(h_output, d_output, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
