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


void memoryInit(el_t *h_table, el_t **d_table, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_table, tableLen * sizeof(**d_table));
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

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_input, &d_table, tableLen);

    startStopwatch(&timer);

    for (uint_t startBit = 0; startBit < sizeof(uint_t) * 8; startBit += BIT_COUNT) {
        runSortBlockKernel(d_table, tableLen, startBit, orderAsc);
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel radix sort.");

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
