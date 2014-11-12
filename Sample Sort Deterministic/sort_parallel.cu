#include <stdio.h>
#include <climits>
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
Initializes DEVICE memory needed for paralel sort implementation.
*/
void memoryInit(el_t *h_input, el_t **d_dataInput, el_t **d_dataBuffer, uint_t tableLen) {
    cudaError_t error;

    // Data memory allocation
    error = cudaMalloc(d_dataInput, tableLen * sizeof(**d_dataInput));
    checkCudaError(error);
    error = cudaMalloc(d_dataBuffer, tableLen * sizeof(**d_dataBuffer));
    checkCudaError(error);

    error = cudaMemcpy(*d_dataInput, h_input, tableLen * sizeof(**d_dataInput), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(el_t *dataTable, uint_t tableLen, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel << <dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable) >> >(
        dataTable, tableLen, sortOrder
        );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

void runPrintTableKernel(el_t *table, uint_t tableLen) {
    printTableKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void sampleSort(el_t *dataTable, el_t *dataBuffer, uint_t tableLen, order_t sortOrder) {
    runBitoicSortKernel(dataTable, tableLen, sortOrder);
}

void sortParallel(el_t *h_dataInput, el_t *h_dataOutput, uint_t tableLen, order_t sortOrder) {
    el_t *d_dataInput, *d_dataBuffer;

    LARGE_INTEGER timer;
    cudaError_t error;

    memoryInit(h_dataInput, &d_dataInput, &d_dataBuffer, tableLen);

    startStopwatch(&timer);
    sampleSort(d_dataInput, d_dataBuffer, tableLen, sortOrder);

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer, "Executing parallel sample sort.");
    printf("Operations (pair swaps): %.2f M/s\n", tableLen / 500.0 / time);

    error = cudaMemcpy(h_dataOutput, d_dataInput, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_dataInput);
}
