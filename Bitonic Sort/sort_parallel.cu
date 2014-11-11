#include <stdio.h>
#include <math.h>
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
Initializes device memory.
*/
void memoryDataInit(el_t *h_table, el_t **d_table, uint_t tableLen) {
    cudaError_t error;

    error = cudaMalloc(d_table, tableLen * sizeof(**d_table));
    checkCudaError(error);
    error = cudaMemcpy(*d_table, h_table, tableLen * sizeof(**d_table), cudaMemcpyHostToDevice);
    checkCudaError(error);
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(el_t *table, uint_t tableLen, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*table)>>>(
        table, tableLen, sortOrder
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

void runBitonicMergeGlobalKernel(el_t *table, uint_t tableLen, uint_t phase, uint_t step, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    dim3 dimGrid(tableLen / (2 * THREADS_PER_GLOBAL_MERGE), 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeGlobalKernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge global kernel");*/
}

void runBitoicMergeKernel(el_t *table, uint_t tableLen, uint_t phase, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    dim3 dimGrid(tableLen / (2 * THREADS_PER_LOCAL_MERGE), 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeLocalKernel<<<dimGrid, dimBlock, 2 * THREADS_PER_LOCAL_MERGE * sizeof(*table)>>>(
        table, phase, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge local kernel");*/
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, order_t sortOrder) {
    el_t *d_table;
    // Number of phases, which can be executed in shared memory (stride is lower than
    // shared memory size)
    int_t phasesSharedMem = log2((double)min(tableLen, 2 * THREADS_PER_LOCAL_MERGE));
    int_t phasesAll = log2((double)tableLen);

    LARGE_INTEGER timer;
    cudaError_t error;

    // Global bitonic merge doesn't use shared memory -> preference can be set for L1
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(bitonicMergeGlobalKernel, cudaFuncCachePreferL1);
    memoryDataInit(h_input, &d_table, tableLen);

    startStopwatch(&timer);
    runBitoicSortKernel(d_table, tableLen, sortOrder);

    /*for (uint_t phase = phasesSharedMem + 1; phase <= phasesAll; phase++) {
        for (int step = phase; step > phasesSharedMem; step--) {
            runBitonicMergeGlobalKernel(d_table, tableLen, phase, step, orderAsc);
        }

        runBitoicMergeKernel(d_table, tableLen, phase, orderAsc);
    }*/

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel bitonic sort.");

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
