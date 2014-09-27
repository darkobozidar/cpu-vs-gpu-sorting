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


void memoryInit(el_t *h_table, el_t **d_table, uint_t **blockOffsets, uint_t **blocksSizes, uint_t tableLen,
                uint_t blocksLen) {
    cudaError_t error;

    error = cudaMalloc(d_table, tableLen * sizeof(**d_table));
    checkCudaError(error);
    error = cudaMalloc(blockOffsets, blocksLen * sizeof(**blockOffsets));
    checkCudaError(error);
    error = cudaMalloc(blocksSizes, blocksLen * sizeof(**blocksSizes));
    checkCudaError(error);

    error = cudaMemcpy(*d_table, h_table, tableLen * sizeof(**d_table), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void runSortBlockKernel(el_t *table, uint_t tableLen, uint_t startBit, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_SORT);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    sortBlockKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*table)>>>(
        table, startBit, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel radix sort of blocks.");*/
}

void runGenerateBlocksKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t tableLen,
                             uint_t startBit) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_SORT);
    uint_t sharedMemSize = 2 * threadBlockSize * sizeof(uint_t) + 2 * (1 << BIT_COUNT) * sizeof(uint_t);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateBlocksKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        table, blockOffsets, blockSizes, startBit
    );
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table;
    uint_t *d_blockOffsets, *d_blockSizes;
    uint_t blocksLen = (1 << BIT_COUNT) * (tableLen / (2 * THREADS_PER_SORT));

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_input, &d_table, &d_blockOffsets, &d_blockSizes, tableLen, blocksLen);

    startStopwatch(&timer);

    /*runSortBlockKernel(d_table, tableLen, 0, orderAsc);
    runGenerateBlocksKernel(d_table, d_blockOffsets, d_blockSizes, tableLen, 0);*/

    for (uint_t startBit = 0; startBit < sizeof(uint_t) * 8; startBit += BIT_COUNT) {
        runSortBlockKernel(d_table, tableLen, startBit, orderAsc);
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel radix sort.");

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
