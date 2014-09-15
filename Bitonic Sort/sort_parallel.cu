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
void runBitoicSortKernel(el_t *table, uint_t tableLen, uint_t subBlockSize, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
        table, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);*/
    //endStopwatch(timer, "Executing bitonic sort kernel");
}

void runBitonicMergeGlobalKernel(el_t *table, uint_t tableLen, uint_t phase, uint_t step, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    dim3 dimGrid(tableLen / (THREADS_PER_GLOBAL_MERGE * 2), 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    startStopwatch(&timer);
    bitonicMergeGlobalKernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);*/
    /*endStopwatch(timer, "Executing multistep kernel");*/
}

void runBitoicMergeKernel(el_t *table, uint_t tableLen, uint_t subBlockSize, uint_t phase, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicMergeKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
        table, phase, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);*/
    //endStopwatch(timer, "Executing bitonic sort kernel");
}

void runPrintTableKernel(el_t *table, uint_t tableLen) {
    printTableKernel<<<1, 1>>>(table, tableLen);
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table;
    // Every thread loads and sorts 2 elements in first bitonic sort kernel
    uint_t subBlockSize = min(tableLen, 2 * THREADS_PER_LOCAL_MERGE);
    int_t phasesAll = log2((double)tableLen);
    int_t phasesSharedMem = log2((double)subBlockSize);

    LARGE_INTEGER timer;
    cudaError_t error;

    // In bitonic global merge kernel no shared memory is used -> preference can be set for L1
    cudaFuncSetCacheConfig(bitonicMergeGlobalKernel, cudaFuncCachePreferL1);
    memoryDataInit(h_input, &d_table, tableLen);

    startStopwatch(&timer);
    runBitoicSortKernel(d_table, tableLen, subBlockSize, orderAsc);

    /*printf("After bitonic sort\n");
    runPrintTableKernel(d_table, tableLen);*/

    for (uint_t phase = phasesSharedMem + 1; phase <= phasesAll; phase++) {
        for (int step = phase; step > phasesSharedMem; step--) {
            runBitonicMergeGlobalKernel(d_table, tableLen, phase, step, orderAsc);
        }
        runBitoicMergeKernel(d_table, tableLen, subBlockSize, phase, orderAsc);

        /*if (phase == 2) {
            printf("After bitonic merge\n");
            runPrintTableKernel(d_table, tableLen);
        }*/
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing parallel bitonic sort.");

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);
}
