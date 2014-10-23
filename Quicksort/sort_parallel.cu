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
void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, lparam_t **d_localParams, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);
    error = cudaMalloc(d_localParams, MAX_SEQUENCES * sizeof(**d_localParams));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

void runQuickSortLocalKernel(el_t *input, el_t *output, lparam_t *localParams, uint_t tableLen, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block-scan array size, array size for bitonic sort)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL * sizeof(uint_t), BITONIC_SORT_SIZE_LOCAL * sizeof(*input)
    );
    uint_t elementsPerBlock = tableLen / MAX_SEQUENCES;
    dim3 dimGrid(MAX_SEQUENCES, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL, 1, 1);

    startStopwatch(&timer);
    quickSortLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        input, output, localParams, tableLen, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing local parallel quicksort.");*/
}

void quickSort(el_t *dataInput, el_t *dataBuffer, lparam_t *h_localParams, lparam_t *d_localParams,
               uint_t tableLen, bool orderAsc) {
    // TODO handle empty sub-blocks
    h_localParams[0].start = 0;
    h_localParams[0].length = 16;
    h_localParams[0].direction = false;
    h_localParams[1].start = 16;
    h_localParams[1].length = 0;
    h_localParams[1].direction = false;

    cudaMemcpy(d_localParams, h_localParams, MAX_SEQUENCES * sizeof(*d_localParams), cudaMemcpyHostToDevice);

    runQuickSortLocalKernel(dataInput, dataBuffer, d_localParams, tableLen, orderAsc);
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, bool orderAsc) {
    el_t *d_dataInput, *d_dataBuffer;
    lparam_t h_localParams[MAX_SEQUENCES], *d_localParams;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_dataInput, &d_dataInput, &d_dataBuffer, &d_localParams, tableLen);

    startStopwatch(&timer);
    quickSort(d_dataInput, d_dataBuffer, h_localParams, d_localParams, tableLen, orderAsc);

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel quicksort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    error = cudaMemcpy(h_dataOutput, d_dataBuffer, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_dataInput);
    cudaFree(d_dataBuffer);
}
