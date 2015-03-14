#ifndef BITONIC_SORT_ADAPTIVE_PARALLEL_H
#define BITONIC_SORT_ADAPTIVE_PARALLEL_H

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
#include "data_types.h"


///*
//Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
//needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
//*/
//void runAddPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder)
//{
//    uint_t tableLenPower2 = nextPowerOf2(tableLen);
//
//    // If table length is already power of 2, than no padding is needed
//    if (tableLen == tableLenPower2)
//    {
//        return;
//    }
//
//    uint_t paddingLength = tableLenPower2 - tableLen;
//
//    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;;
//    dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_PADDING, 1, 1);
//
//    // Depending on sort order different value is used for padding.
//    if (sortOrder == ORDER_ASC)
//    {
//        addPaddingKernel<MAX_VAL> << <dimGrid, dimBlock >> >(dataTable, dataBuffer, tableLen, paddingLength);
//    }
//    else
//    {
//        addPaddingKernel<MIN_VAL> << <dimGrid, dimBlock >> >(dataTable, dataBuffer, tableLen, paddingLength);
//    }
//}
//
///*
//Sorts sub-blocks of input data with bitonic sort.
//*/
//void runBitoicSortKernel(data_t *keys, data_t *values, uint_t tableLen, order_t sortOrder)
//{
//    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//    // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
//    // necessary for entire padded table to be ordered. It is only necessary that table is ordered to the next
//    // multiple of number of elements processed by one thread block. This ensures that bitonic sequences get
//    // created for entire original table length (padded elemens are MIN/MAX values and sort would't change
//    // anything).
//    uint_t tableLenRoundedUp;
//    if (tableLen > elemsPerThreadBlock)
//    {
//        tableLenRoundedUp = roundUp(tableLen, elemsPerThreadBlock);
//    }
//    // For sequences shorter than "tableLenRoundedUp" only bitonic sort kernel is needed to sort them (whithout
//    // any other kernels). In that case table size can be rounded to next power of 2.
//    else
//    {
//        tableLenRoundedUp = nextPowerOf2(tableLen);
//    }
//
//    // "2 *" is needed because keys AND values are sorted in shared memory
//    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*keys);
//    dim3 dimGrid((tableLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);
//
//    if (sortOrder == ORDER_ASC)
//    {
//        bitonicSortKernel<ORDER_ASC> << <dimGrid, dimBlock, sharedMemSize >> >(keys, values, tableLenRoundedUp);
//    }
//    else
//    {
//        bitonicSortKernel<ORDER_DESC> << <dimGrid, dimBlock, sharedMemSize >> >(keys, values, tableLenRoundedUp);
//    }
//}
//
///*
//Initializes intervals and continues to evolve them until the end step.
//*/
//void runInitIntervalsKernel(
//    data_t *dataTable, interval_t *intervals, uint_t tableLen, uint_t phasesAll, uint_t stepStart,
//    uint_t stepEnd, order_t sortOrder
//    )
//{
//    // How many intervals have to be generated
//    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
//    uint_t threadBlockSize = min((intervalsLen - 1) / ELEMS_PER_INIT_INTERVALS + 1, THREADS_PER_INIT_INTERVALS);
//
//    // "2 *" because of BUFFER MEMORY for intervals
//    uint_t sharedMemSize = 2 * ELEMS_PER_INIT_INTERVALS * threadBlockSize * sizeof(*intervals);
//    dim3 dimGrid((intervalsLen - 1) / (ELEMS_PER_INIT_INTERVALS * threadBlockSize) + 1, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    if (sortOrder == ORDER_ASC)
//    {
//        initIntervalsKernel<ORDER_ASC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            dataTable, intervals, tableLen, stepStart, stepEnd
//            );
//    }
//    else
//    {
//        initIntervalsKernel<ORDER_DESC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            dataTable, intervals, tableLen, stepStart, stepEnd
//            );
//    }
//}
//
///*
//Evolves intervals from start step to end step.
//*/
//void runGenerateIntervalsKernel(
//    data_t *table, interval_t *inputIntervals, interval_t *outputIntervals, uint_t tableLen, uint_t phasesAll,
//    uint_t phase, uint_t stepStart, uint_t stepEnd, order_t sortOrder
//    )
//{
//    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
//    uint_t threadBlockSize = min((intervalsLen - 1) / ELEMS_PER_GEN_INTERVALS + 1, THREADS_PER_GEN_INTERVALS);
//
//    // "2 *" because of BUFFER MEMORY for intervals
//    uint_t sharedMemSize = 2 * ELEMS_PER_GEN_INTERVALS * threadBlockSize * sizeof(*inputIntervals);
//    dim3 dimGrid((intervalsLen - 1) / (ELEMS_PER_GEN_INTERVALS * threadBlockSize) + 1, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    if (sortOrder == ORDER_ASC)
//    {
//        generateIntervalsKernel<ORDER_ASC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            table, inputIntervals, outputIntervals, tableLen, phase, stepStart, stepEnd
//            );
//    }
//    else
//    {
//        generateIntervalsKernel<ORDER_DESC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            table, inputIntervals, outputIntervals, tableLen, phase, stepStart, stepEnd
//            );
//    }
//}
//
///*
//Runs kernel, whic performs bitonic merge from provided intervals.
//*/
//void runBitoicMergeKernel(
//    data_t *keys, data_t *values, data_t *keysBuffer, data_t *valuesBuffer, interval_t *intervals,
//    uint_t tableLen, uint_t phasesBitonicMerge, uint_t phase, order_t sortOrder
//    )
//{
//    uint_t elemsPerThreadBlock = THREADS_PER_MERGE * ELEMS_PER_MERGE;
//    // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
//    // necessary for entire padded table to be merged. It is only necessary that table is merged to the next
//    // multiple of phase stride.
//    uint_t tableLenRoundedUp = roundUp(tableLen, 1 << phase);
//
//    // "2 *" is needed because keys AND values are marged in shared memory
//    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*keys);
//    dim3 dimGrid((tableLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
//    dim3 dimBlock(THREADS_PER_MERGE, 1, 1);
//
//    if (sortOrder == ORDER_ASC)
//    {
//        bitonicMergeKernel<ORDER_ASC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            keys, values, keysBuffer, valuesBuffer, intervals, phase
//            );
//    }
//    else
//    {
//        bitonicMergeKernel<ORDER_DESC> << <dimGrid, dimBlock, sharedMemSize >> >(
//            keys, values, keysBuffer, valuesBuffer, intervals, phase
//            );
//    }
//}
//
//double sortParallel(
//    data_t *h_keys, data_t *h_values, data_t *d_keys, data_t *d_values, data_t *d_keysBuffer,
//    data_t *d_valuesBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, uint_t tableLen,
//    order_t sortOrder
//    )
//{
//    uint_t tableLenPower2 = nextPowerOf2(tableLen);
//    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
//
//    // Every thread loads and processes 2 elements
//    uint_t phasesAll = log2((double)tableLenPower2);
//    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
//    uint_t phasesBitonicMerge = log2((double)(THREADS_PER_MERGE * ELEMS_PER_MERGE));
//    uint_t phasesInitIntervals = log2((double)THREADS_PER_INIT_INTERVALS * ELEMS_PER_INIT_INTERVALS);
//    uint_t phasesGenerateIntervals = log2((double)THREADS_PER_GEN_INTERVALS * ELEMS_PER_GEN_INTERVALS);
//
//    LARGE_INTEGER timer;
//    cudaError_t error;
//
//    startStopwatch(&timer);
//    runAddPaddingKernel(d_keys, d_keysBuffer, tableLen, sortOrder);
//    runBitoicSortKernel(d_keys, d_values, tableLen, sortOrder);
//
//    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
//    {
//        uint_t stepStart = phase;
//        uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
//        runInitIntervalsKernel(
//            d_keys, d_intervals, tableLenPower2, phasesAll, stepStart, stepEnd, sortOrder
//            );
//
//        // After initial intervals were generated intervals have to be evolved to the end step
//        while (stepEnd > phasesBitonicMerge)
//        {
//            interval_t *tempIntervals = d_intervals;
//            d_intervals = d_intervalsBuffer;
//            d_intervalsBuffer = tempIntervals;
//
//            stepStart = stepEnd;
//            stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
//            runGenerateIntervalsKernel(
//                d_keys, d_intervalsBuffer, d_intervals, tableLenPower2, phasesAll, phase, stepStart,
//                stepEnd, sortOrder
//                );
//        }
//
//        // Global merge with intervals
//        runBitoicMergeKernel(
//            d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_intervals, tableLen, phasesBitonicMerge,
//            phase, sortOrder
//            );
//
//        // Exchanges keys
//        data_t *tempTable = d_keys;
//        d_keys = d_keysBuffer;
//        d_keysBuffer = tempTable;
//        // Exchanges values
//        tempTable = d_values;
//        d_values = d_valuesBuffer;
//        d_valuesBuffer = tempTable;
//    }
//
//    error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    double time = endStopwatch(timer);
//
//    error = cudaMemcpy(h_keys, d_keys, tableLen * sizeof(*h_keys), cudaMemcpyDeviceToHost);
//    checkCudaError(error);
//    error = cudaMemcpy(h_values, d_values, tableLen * sizeof(*h_values), cudaMemcpyDeviceToHost);
//    checkCudaError(error);
//
//    return time;
//}


#endif
