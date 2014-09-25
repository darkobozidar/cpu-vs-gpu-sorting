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
void memoryDataInit(el_t *h_table, el_t **d_table, el_t **d_buffer, interval_t **intervals,
    interval_t **intervalsBuffer, uint_t tableLen, uint_t intervalsLen) {
    cudaError_t error;

    error = cudaMalloc(d_table, tableLen * sizeof(**d_table));
    checkCudaError(error);
    error = cudaMalloc(d_buffer, tableLen * sizeof(**d_buffer));
    checkCudaError(error);
    error = cudaMalloc(intervals, intervalsLen * sizeof(**intervals));
    checkCudaError(error);
    error = cudaMalloc(intervalsBuffer, intervalsLen * sizeof(**intervalsBuffer));
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
    uint_t subBlockSize = 1 << phasesBitonicSort;
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

void runInitIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t phasesAll,
                            uint_t stepStart, uint_t stepEnd) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_INIT_INTERVALS);
    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    initIntervalsKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*intervals)>>>(
        table, intervals, tableLen, stepStart, stepEnd
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing kernel for initializing intervals");*/
}

void runGenerateIntervalsKernel(el_t *table, interval_t *input, interval_t *output, uint_t tableLen,
                                uint_t phasesAll, uint_t phase, uint_t stepStart, uint_t stepEnd) {
    cudaError_t error;
    LARGE_INTEGER timer;

    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_GEN_INTERVALS);
    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    startStopwatch(&timer);
    generateIntervalsKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*input)>>>(
        table, input, output, tableLen, phase, stepStart, stepEnd
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing kernel for generating intervals");*/
}

void runBitoicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t tableLen,
                          uint_t phasesBitonicMerge, uint_t phase, bool orderAsc) {
    cudaError_t error;
    LARGE_INTEGER timer;

    // Every thread loads and sorts 2 elements
    uint_t subBlockSize = 1 << phasesBitonicMerge;
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    startStopwatch(&timer);
    bitonicMergeKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*input)>>>(
        input, output, intervals, phase, orderAsc
    );
    /*error = cudaDeviceSynchronize();
    checkCudaError(error);
    endStopwatch(timer, "Executing bitonic merge kernel");*/
}

void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    el_t *d_table, *d_buffer;
    interval_t *d_intervals, *d_intervalsBuffer;
    // Every thread loads and sorts 2 elements in first bitonic sort kernel
    uint_t phasesAll = log2((double)tableLen);
    uint_t phasesBitonicSort = log2((double)min(tableLen, 2 * THREADS_PER_SORT));
    uint_t phasesBitonicMerge = log2((double)2 * THREADS_PER_MERGE);
    uint_t phasesInitIntervals = log2((double)2 * THREADS_PER_INIT_INTERVALS);
    uint_t phasesGenerateIntervals = log2((double)2 * THREADS_PER_GEN_INTERVALS);
    uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    LARGE_INTEGER timer;
    double time;
    cudaError_t error;

    memoryDataInit(h_input, &d_table, &d_buffer, &d_intervals, &d_intervalsBuffer, tableLen, intervalsLen);

    startStopwatch(&timer);
    runBitoicSortKernel(d_table, tableLen, phasesBitonicSort, orderAsc);
    //runPrintTableKernel(d_table, tableLen);

    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
        uint_t stepStart = phase;
        uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
        runInitIntervalsKernel(d_table, d_intervals, tableLen, phasesAll, stepStart, stepEnd);

        while (stepEnd > phasesBitonicMerge) {
            interval_t *tempIntervals = d_intervals;
            d_intervals = d_intervalsBuffer;
            d_intervalsBuffer = tempIntervals;

            stepStart = stepEnd;
            stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
            runGenerateIntervalsKernel(d_table, d_intervalsBuffer, d_intervals, tableLen, phasesAll, phase,
                                       stepStart, stepEnd);
        }

        runBitoicMergeKernel(d_table, d_buffer, d_intervals, tableLen, phasesBitonicMerge, phase, orderAsc);
        //runPrintTableKernel(d_table, tableLen);

        el_t *tempTable = d_table;
        d_table = d_buffer;
        d_buffer = tempTable;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    time = endStopwatch(timer, "Executing parallel bitonic sort.");
    printf("Operations: %.2f M/s\n", tableLen / 1000.0 / time);

    error = cudaMemcpy(h_output, d_table, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    cudaFree(d_table);
}
