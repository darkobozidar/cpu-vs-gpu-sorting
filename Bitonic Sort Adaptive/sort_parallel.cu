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


///*
//Sorts sub-blocks of input data with bitonic sort.
//*/
//void runBitoicSortKernel(el_t *table, uint_t tableLen, uint_t phasesBitonicSort, bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    // Every thread loads and sorts 2 elements
//    uint_t subBlockSize = 1 << phasesBitonicSort;
//    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
//    dim3 dimBlock(subBlockSize / 2, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicSortKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
//        table, orderAsc
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic sort kernel");*/
//}
//
//void runInitIntervalsKernel(el_t *table, interval_t *intervals, uint_t tableLen, uint_t phasesAll,
//                            uint_t stepStart, uint_t stepEnd) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
//    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_INIT_INTERVALS);
//    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    initIntervalsKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*intervals)>>>(
//        table, intervals, tableLen, stepStart, stepEnd
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for initializing intervals");*/
//}
//
//void runGenerateIntervalsKernel(el_t *table, interval_t *input, interval_t *output, uint_t tableLen,
//                                uint_t phasesAll, uint_t phase, uint_t stepStart, uint_t stepEnd) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
//    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_GEN_INTERVALS);
//    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    generateIntervalsKernel<<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*input)>>>(
//        table, input, output, tableLen, phase, stepStart, stepEnd
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing kernel for generating intervals");*/
//}
//
//void runBitoicMergeKernel(el_t *input, el_t *output, interval_t *intervals, uint_t tableLen,
//                          uint_t phasesBitonicMerge, uint_t phase, bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    // Every thread loads and sorts 2 elements
//    uint_t phases = min(phasesBitonicMerge, phase);
//    uint_t subBlockSize = 1 << phases;
//    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
//    dim3 dimBlock(subBlockSize / 2, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicMergeKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*input)>>>(
//        input, output, intervals, phase, orderAsc
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic merge kernel");*/
//}

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, uint_t tableLen, order_t sortOrder
)
{
    interval_t *d_intervals, *d_intervalsBuffer;
    // Every thread loads and processes 2 elements
    uint_t phasesAll = log2((double)tableLen);
    uint_t phasesBitonicSort = log2((double)min(tableLen, 2 * THREADS_PER_SORT));
    uint_t phasesBitonicMerge = log2((double)2 * THREADS_PER_MERGE);
    uint_t phasesInitIntervals = log2((double)2 * THREADS_PER_INIT_INTERVALS);
    uint_t phasesGenerateIntervals = log2((double)2 * THREADS_PER_GEN_INTERVALS);
    uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    /*runBitoicSortKernel(d_table, tableLen, phasesBitonicSort, orderAsc);*/

    //for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++) {
    //    uint_t stepStart = phase;
    //    uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
    //    runInitIntervalsKernel(d_table, d_intervals, tableLen, phasesAll, stepStart, stepEnd);

    //    // After initial intervals were generated intervals have to be evolved to the end
    //    while (stepEnd > phasesBitonicMerge) {
    //        interval_t *tempIntervals = d_intervals;
    //        d_intervals = d_intervalsBuffer;
    //        d_intervalsBuffer = tempIntervals;

    //        stepStart = stepEnd;
    //        stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
    //        runGenerateIntervalsKernel(d_table, d_intervalsBuffer, d_intervals, tableLen, phasesAll, phase,
    //                                   stepStart, stepEnd);
    //    }

    //    // Global merge with intervals
    //    runBitoicMergeKernel(d_table, d_buffer, d_intervals, tableLen, phasesBitonicMerge, phase, orderAsc);

    //    el_t *tempTable = d_table;
    //    d_table = d_buffer;
    //    d_buffer = tempTable;
    //}

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
