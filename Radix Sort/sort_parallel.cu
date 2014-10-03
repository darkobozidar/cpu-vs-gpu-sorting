#include <stdio.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"
#include "kernels.h"


/*
Initializes memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_table, el_t **d_table, el_t **d_bufffer, uint_t **d_bucketOffsetsGlobal,
                uint_t **bucketOffsetsLocal, uint_t **d_bucketSizes, uint_t tableLen, uint_t bucketsLen) {
    cudaError_t error;

    error = cudaMalloc(d_table, tableLen * sizeof(**d_table));
    checkCudaError(error);
    error = cudaMalloc(d_bufffer, tableLen * sizeof(**d_bufffer));
    checkCudaError(error);
    error = cudaMalloc(d_bucketOffsetsGlobal, bucketsLen * sizeof(**bucketOffsetsLocal));
    checkCudaError(error);
    error = cudaMalloc(bucketOffsetsLocal, bucketsLen * sizeof(**bucketOffsetsLocal));
    checkCudaError(error);
    error = cudaMalloc(d_bucketSizes, bucketsLen * sizeof(**d_bucketSizes));
    checkCudaError(error);

    error = cudaMemcpy(*d_table, h_table, tableLen * sizeof(**d_table), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Initializes library CUDPP, which implements scan() function
*/
void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen) {
    // Initializes the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    *scanPlan = 0;
    CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, tableLen, 1, 0);

    if (result != CUDPP_SUCCESS) {
        printf("Error creating CUDPPPlan\n");
        getchar();
        exit(-1);
    }
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

void runRadixSortGlobalKernel(el_t *input,  el_t *output, uint_t *offsetsLocal, uint_t *offsetsGlobal,
                              uint_t tableLen, uint_t bitOffset, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t threadBlockSize = min(tableLen / 2, THREADS_PER_GLOBAL_SORT);
    dim3 dimGrid(tableLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    radixSortGlobalKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*input)>>>(
        input, output, offsetsLocal, offsetsGlobal, bitOffset
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing global parallel radix sort.");*/
}

void runPrintTableKernel(uint_t *table, uint_t tableLen) {
    printTableKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table, *d_bufffer;
    uint_t *d_bucketOffsetsLocal, *d_bucketOffsetsGlobal, *d_bucketSizes;
    uint_t threadsPerSort = min(tableLen / 2, THREADS_PER_LOCAL_SORT);
    uint_t bucketsLen = RADIX * (tableLen / (2 * threadsPerSort));
    CUDPPHandle scanPlan;

    LARGE_INTEGER timer;
    cudaError_t error;

    // Init memory and library CUDPP
    memoryInit(h_input, &d_table, &d_bufffer, &d_bucketOffsetsLocal, &d_bucketOffsetsGlobal, &d_bucketSizes,
               tableLen, bucketsLen);
    cudppInitScan(&scanPlan, bucketsLen);

    startStopwatch(&timer);

    /*runRadixSortLocalKernel(d_table, tableLen, 0, orderAsc);*/

    for (uint_t bitOffset = 0; bitOffset < sizeof(uint_t) * 8; bitOffset += BIT_COUNT) {
        runRadixSortLocalKernel(d_table, tableLen, bitOffset, orderAsc);
        runGenerateBucketsKernel(d_table, d_bucketOffsetsLocal, d_bucketSizes, tableLen, bitOffset);

        CUDPPResult result = cudppScan(scanPlan, d_bucketOffsetsGlobal, d_bucketSizes, bucketsLen);
        if (result != CUDPP_SUCCESS) {
            printf("Error in cudppScan()\n");
            getchar();
            exit(-1);
        }

        runRadixSortGlobalKernel(
            d_table, d_bufffer, d_bucketOffsetsLocal, d_bucketOffsetsGlobal, tableLen, bitOffset, orderAsc
        );

        el_t *temp = d_table;
        d_table = d_bufffer;
        d_bufffer = temp;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel radix sort.");

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_table);
    cudaFree(d_bufffer);
}
