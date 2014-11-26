#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "kernels.h"


/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
        dataTable, tableLen, sortOrder
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

void runBitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeGlobalKernel<<<dimGrid, dimBlock>>>(dataTable, tableLen, step, step == phase, sortOrder);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge global kernel");*/
}

void runBitoicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeLocalKernel<<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
        dataTable, tableLen, step, phase == step, sortOrder
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge local kernel");*/
}

void runPrintTableKernel(data_t *table, uint_t tableLen) {
    printTableKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void sortParallel(data_t *h_input, data_t *h_output, data_t *d_dataTable, uint_t tableLen, order_t sortOrder) {
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));
    uint_t phasesAll = log2((double)tableLenPower2);

    LARGE_INTEGER timer;
    cudaError_t error;

    // Global bitonic merge doesn't use shared memory -> preference can be set for L1
    // TODO test
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(bitonicMergeGlobalKernel, cudaFuncCachePreferL1);

    startStopwatch(&timer);
    runBitoicSortKernel(d_dataTable, tableLen, sortOrder);

    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
        uint_t step = phase;
        while (step > phasesMergeLocal) {
            runBitonicMergeGlobalKernel(d_dataTable, tableLen, phase, step, sortOrder);
            step--;
        }

        runBitoicMergeLocalKernel(d_dataTable, tableLen, phase, step, sortOrder);
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);

    double time = endStopwatch(timer);
    printf("Parallel: %.5lf ms. Swaps/s: %.2f M/s\n", time, tableLen / 500.0 / time);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
