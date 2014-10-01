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
void memoryInit(el_t *h_table, el_t **d_input, el_t **d_output, uint_t **d_bucketOffsetsGlobal,
                uint_t **bucketOffsetsLocal, uint_t **d_bucketSizes, uint_t tableLen, uint_t bucketsLen) {
    cudaError_t error;

    error = cudaMalloc(d_input, tableLen * sizeof(**d_input));
    checkCudaError(error);
    error = cudaMalloc(d_output, tableLen * sizeof(**d_output));
    checkCudaError(error);
    error = cudaMalloc(d_bucketOffsetsGlobal, bucketsLen * sizeof(**bucketOffsetsLocal));
    checkCudaError(error);
    error = cudaMalloc(bucketOffsetsLocal, bucketsLen * sizeof(**bucketOffsetsLocal));
    checkCudaError(error);
    error = cudaMalloc(d_bucketSizes, bucketsLen * sizeof(**d_bucketSizes));
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

void runGenerateBucketsKernel(el_t *table, uint_t *blockOffsets, uint_t *blockSizes, uint_t tableLen,
                              uint_t bitOffset) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_LOCAL_SORT);
    uint_t sharedMemSize = 2 * threadBlockSize * sizeof(uint_t) + 2 * RADIX * sizeof(uint_t);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateBucketsKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
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
    uint_t *d_bucketOffsetsLocal, *d_bucketOffsetsGlobal, *d_bucketSizes;
    uint_t bucketsLen = RADIX * (tableLen / (2 * THREADS_PER_LOCAL_SORT));

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_input, &d_input, &d_output, &d_bucketOffsetsLocal, &d_bucketOffsetsGlobal, &d_bucketSizes,
               tableLen, bucketsLen);

    startStopwatch(&timer);

    runRadixSortLocalKernel(d_input, tableLen, 0, orderAsc);
    runGenerateBucketsKernel(d_input, d_bucketOffsetsLocal, d_bucketSizes, tableLen, 0);

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
