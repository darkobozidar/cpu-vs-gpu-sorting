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
#include "kernels.h"


/*
Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
*/
void runAddPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);

    // If table length is already power of 2, than no padding is needed
    if (tableLen == tableLenPower2)
    {
        return;
    }

    uint_t paddingLength = tableLenPower2 - tableLen;

    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;;
    dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_PADDING, 1, 1);

    // Depending on sort order different value is used for padding.
    if (sortOrder == ORDER_ASC)
    {
        addPaddingKernel<MAX_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
    else
    {
        addPaddingKernel<MIN_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
    // necessary for entire padded table to be ordered. It is only necessary that table is ordered to the next
    // multiple of number of elements processed by one thread block. This ensures that bitonic sequences get
    // created for entire original table length (padded elemens are MIN/MAX values and sort would't change
    // anything).
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerThreadBlock);

    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*dataTable);
    dim3 dimGrid((tableLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable, tableLenRoundedUp);
    }
    else
    {
        bitonicSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable, tableLenRoundedUp);
    }
}

void runInitIntervalsKernel(
    data_t *table, interval_t *intervals, uint_t tableLen, uint_t phasesAll, uint_t stepStart, uint_t stepEnd,
    order_t sortOrder
)
{
    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_INIT_INTERVALS);
    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        initIntervalsKernel<ORDER_ASC><<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*intervals)>>>(
            table, intervals, tableLen, stepStart, stepEnd
        );
    }
    else
    {
        initIntervalsKernel<ORDER_DESC><<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*intervals)>>>(
            table, intervals, tableLen, stepStart, stepEnd
        );
    }
}

void runGenerateIntervalsKernel(
    data_t *table, interval_t *input, interval_t *output, uint_t tableLen, uint_t phasesAll, uint_t phase,
    uint_t stepStart, uint_t stepEnd, order_t sortOrder
)
{
    uint_t intervalsLen = 1 << (phasesAll - stepEnd);
    uint_t threadBlockSize = min(intervalsLen / 2, THREADS_PER_GEN_INTERVALS);
    dim3 dimGrid(intervalsLen / (2 * threadBlockSize), 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        generateIntervalsKernel<ORDER_ASC><<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*input)>>>(
            table, input, output, tableLen, phase, stepStart, stepEnd
        );
    }
    else
    {
        generateIntervalsKernel<ORDER_DESC><<<dimGrid, dimBlock, 2 * threadBlockSize * sizeof(*input)>>>(
            table, input, output, tableLen, phase, stepStart, stepEnd
        );
    }
}

void runBitoicMergeKernel(
    data_t *input, data_t *output, interval_t *intervals, uint_t tableLen, uint_t phasesBitonicMerge,
    uint_t phase, order_t sortOrder
)
{
    // Every thread loads and sorts 2 elements
    uint_t phases = min(phasesBitonicMerge, phase);
    uint_t subBlockSize = 1 << phases;
    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
    dim3 dimBlock(subBlockSize / 2, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicMergeKernel<ORDER_ASC><<<dimGrid, dimBlock, subBlockSize * sizeof(*input)>>>(
            input, output, intervals, phase
        );
    }
    else
    {
        bitonicMergeKernel<ORDER_DESC><<<dimGrid, dimBlock, subBlockSize * sizeof(*input)>>>(
            input, output, intervals, phase
        );
    }
}

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, interval_t *d_intervals,
    interval_t *d_intervalsBuffer, uint_t tableLen, order_t sortOrder
)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;

    // Every thread loads and processes 2 elements
    uint_t phasesAll = log2((double)tableLenPower2);
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    uint_t phasesBitonicMerge = log2((double)2 * THREADS_PER_MERGE);
    uint_t phasesInitIntervals = log2((double)2 * THREADS_PER_INIT_INTERVALS);
    uint_t phasesGenerateIntervals = log2((double)2 * THREADS_PER_GEN_INTERVALS);
    uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    runAddPaddingKernel(d_dataTable, d_dataBuffer, tableLen, sortOrder);
    runBitoicSortKernel(d_dataTable, tableLen, sortOrder);

    for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        uint_t stepStart = phase;
        uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
        runInitIntervalsKernel(
            d_dataTable, d_intervals, tableLenPower2, phasesAll, stepStart, stepEnd, sortOrder
        );

        // After initial intervals were generated intervals have to be evolved to the end
        while (stepEnd > phasesBitonicMerge)
        {
            interval_t *tempIntervals = d_intervals;
            d_intervals = d_intervalsBuffer;
            d_intervalsBuffer = tempIntervals;

            stepStart = stepEnd;
            stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
            runGenerateIntervalsKernel(
                d_dataTable, d_intervalsBuffer, d_intervals, tableLenPower2, phasesAll, phase, stepStart,
                stepEnd, sortOrder
            );
        }

        // Global merge with intervals
        runBitoicMergeKernel(
            d_dataTable, d_dataBuffer, d_intervals, tableLenPower2, phasesBitonicMerge, phase, sortOrder
        );

        data_t *tempTable = d_dataTable;
        d_dataTable = d_dataBuffer;
        d_dataBuffer = tempTable;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
