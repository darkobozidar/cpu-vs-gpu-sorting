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
Initializes memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_table, el_t **d_input, el_t **d_output, uint_t **d_blockOffsets, uint_t **d_blocksSizes,
                uint_t tableLen, uint_t blocksLen) {
    cudaError_t error;

    error = cudaMalloc(d_input, tableLen * sizeof(**d_input));
    checkCudaError(error);
    error = cudaMalloc(d_output, tableLen * sizeof(**d_output));
    checkCudaError(error);
    error = cudaMalloc(d_blockOffsets, blocksLen * sizeof(**d_blockOffsets));
    checkCudaError(error);
    error = cudaMalloc(d_blocksSizes, blocksLen * sizeof(**d_blocksSizes));
    checkCudaError(error);

    error = cudaMemcpy(*d_input, h_table, tableLen * sizeof(**d_input), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Runs kernel, which sorts data blocks in shared memory with radix sort.
*/
void runRadixSortLocalKernel(el_t *table, uint_t tableLen, uint_t bitOffset, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_LOCAL_SORT);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    radixSortLocalKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*table)>>>(
        table, bitOffset, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing local parallel radix sort.");*/
}

void runGenerateBlocksKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t tableLen,
                             uint_t bitOffset) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_BLOCK_GEN);
    uint_t sharedMemSize = 2 * threadBlockSize * sizeof(uint_t) + 2 * (1 << BIT_COUNT) * sizeof(uint_t);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateBlocksKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        table, blockOffsets, blockSizes, bitOffset
    );
}

void runPrintTableKernel(el_t *table, uint_t tableLen) {
    printTableKernel << <1, 1 >> >(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_input, *d_output;
    uint_t *d_blockOffsets, *d_blockSizes;
    uint_t blocksLen = (1 << BIT_COUNT) * (tableLen / (2 * THREADS_PER_BLOCK_GEN));

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_input, &d_input, &d_output, &d_blockOffsets, &d_blockSizes, tableLen, blocksLen);

    startStopwatch(&timer);

    runRadixSortLocalKernel(d_input, tableLen, 0, orderAsc);
    /*runGenerateBlocksKernel(d_input, d_blockOffsets, d_blockSizes, tableLen, 0);*/

    /*for (uint_t bitOffset = 0; bitOffset < sizeof(uint_t) * 8; bitOffset += BIT_COUNT) {
        runSortBlockKernel(d_input, tableLen, bitOffset, orderAsc);
    }*/

    el_t *temp = d_input;
    d_input = d_output;
    d_output = temp;

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel radix sort.");

    error = cudaMemcpy(h_output, d_output, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
