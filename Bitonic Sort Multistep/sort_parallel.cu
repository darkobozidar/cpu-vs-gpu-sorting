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
void runBitoicSortKernel(el_t *table, uint_t tableLen, uint_t phasesBitonicSort, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t subBlockSize = 1 << (phasesBitonicSort + 1);
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicSortKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
        table, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

/*
Runs multistep kernel, which uses registers.
*/
void runMultiStepKernel(el_t *table, uint_t tableLen, uint_t phase, uint_t step, uint_t degree,
                                 bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t partitionSize = tableLen / (1 << degree);
    uint_t maxThreadBlockSize = MAX_THREADS_PER_MULTISTEP;
    uint_t threadBlockSize = min(partitionSize, maxThreadBlockSize);
    dim3 dimGrid(partitionSize / threadBlockSize, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    if (degree == 1) {
        multiStep1Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    } else if (degree == 2) {
        multiStep2Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    } else if (degree == 3) {
        multiStep3Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    } else if (degree == 4) {
        multiStep4Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
    }
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing multistep kernel using registers");*/
}

void runBitoicMergeKernel(el_t *table, uint_t tableLen, uint_t phasesBitonicMerge, uint_t phase, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t subBlockSize = 1 << (phasesBitonicMerge + 1);
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicMergeKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
        table, phase, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic sort kernel");*/
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table;
    // Every thread loads and sorts 2 elements in first bitonic sort kernel
    int_t phasesAll = log2((double)tableLen);
    int_t phasesBitonicSort = log2((double)min(tableLen / 2, THREADS_PER_SORT));
    int_t phasesBitonicMerge = log2((double)THREADS_PER_MERGE);

    LARGE_INTEGER timer;
    double time;
    cudaError_t error;

    memoryDataInit(h_input, &d_table, tableLen);

    startStopwatch(&timer);
    runBitoicSortKernel(d_table, tableLen, phasesBitonicSort, orderAsc);

    for (uint_t phase = phasesBitonicSort + 2; phase <= phasesAll; phase++) {
        int_t step = phase;

        // Multisteps
        for (uint_t degree = MAX_MULTI_STEP; degree > 0; degree--) {
            for (; step > phasesBitonicMerge + degree; step -= degree) {
                runMultiStepKernel(d_table, tableLen, phase, step, degree, orderAsc);
            }
        }

        runBitoicMergeKernel(d_table, tableLen, phasesBitonicMerge, phase, orderAsc);
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    time = endStopwatch(timer, "Executing parallel bitonic sort.");
    printf("Operations: %.2f M/s\n", tableLen / 1000 / time);

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_table);
}
