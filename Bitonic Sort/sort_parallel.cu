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
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    bitonicSortKernel<<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
        dataTable, tableLen, sortOrder
    );
}

/*
Merges array, if data blocks are larger than shared memory size. It executes only of STEP on PHASE per kernel lounch.
*/
void runBitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_MERGE, 1, 1);

    bitonicMergeGlobalKernel<<<dimGrid, dimBlock>>>(dataTable, tableLen, step, step == phase, sortOrder);
}

/*
Merges array when stride is lower than shared memory size. It executes all remaining STEPS of current PHASE.
*/
void runBitoicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t phase, uint_t step, order_t sortOrder)
{
    // Every thread loads and sorts 2 elements
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_MERGE, 1, 1);

    bitonicMergeLocalKernel<<<dimGrid, dimBlock, elemsPerThreadBlock * sizeof(*dataTable)>>>(
        dataTable, tableLen, step, phase == step, sortOrder
    );
}

/*
Sorts data with NORMALIZED BITONIC SORT.
*/
double sortParallel(data_t *h_input, data_t *h_output, data_t *d_dataTable, uint_t tableLen, order_t sortOrder) {
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
    cudaFuncSetCacheConfig(bitonicMergeGlobalKernel, cudaFuncCachePreferL1);

    startStopwatch(&timer);
    runBitoicSortKernel(d_dataTable, tableLen, sortOrder);

    // Bitonic merge
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

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
