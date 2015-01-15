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
Executes kernel for finding min/max values. Every thread block searches for min/max values in their
corresponding chunk of data. This means kernel will return a list of min/max values with same length
as number of thread blocks executing in kernel.
*/
uint_t runMinMaxReductionKernel(data_t *primaryArray, data_t *bufferArray, uint_t tableLen)
{
    // Half of the array for min values and the other half for max values
    uint_t sharedMemSize = 2 * THREADS_PER_REDUCTION * sizeof(data_t);
    dim3 dimGrid((tableLen - 1) / (THREADS_PER_REDUCTION * ELEMENTS_PER_THREAD_REDUCTION) + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_REDUCTION, 1, 1);

    minMaxReductionKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        primaryArray, bufferArray, tableLen
    );

    return dimGrid.x;
}

/*
Searches for min/max values in array.
*/
void minMaxReduction(
    data_t *h_dataInput, data_t *d_dataInput, data_t *d_dataBuffer, data_t *h_minMaxValues, uint_t tableLen,
    data_t &minVal, data_t &maxVal
)
{
    minVal = MAX_VAL;
    maxVal = MIN_VAL;

    // Checks whether array is short enough to be reduced entirely on host or if reduction on device is needed
    if (tableLen > THRESHOLD_REDUCTION)
    {
        // Kernel returns array with min/max values of length numVales
        uint_t numValues = runMinMaxReductionKernel(d_dataInput, d_dataBuffer, tableLen);

        cudaError_t error = cudaMemcpy(
            h_minMaxValues, d_dataBuffer, 2 * numValues * sizeof(*h_minMaxValues), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);

        data_t *minValues = h_minMaxValues;
        data_t *maxValues = h_minMaxValues + numValues;

        // Finnishes reduction on host
        for (uint_t i = 0; i < numValues; i++)
        {
            minVal = min(minVal, minValues[i]);
            maxVal = max(maxVal, maxValues[i]);
        }
    }
    else
    {
        for (uint_t i = 0; i < tableLen; i++)
        {
            minVal = min(minVal, h_dataInput[i]);
            maxVal = max(maxVal, h_dataInput[i]);
        }
    }
}

/*
Runs global (multiple thread blocks process one sequence) quicksort and coppies required data to and
from device.
*/
void runQuickSortGlobalKernel(
    data_t *dataTable, data_t *dataBuffer, d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev,
    uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, uint_t numSeqGlobal, uint_t threadBlockCounter
)
{
    cudaError_t error;

    // 1. arg: Size of array for calculation of min/max value ("2" because of MIN and MAX)
    // 2. arg: Size of array needed to perform scan of counters for number of elements lower/greater than
    //         pivot ("2" because of intra-warp scan).
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_GLOBAL * sizeof(data_t), 2 * THREADS_PER_SORT_GLOBAL * sizeof(uint_t)
    );
    dim3 dimGrid(threadBlockCounter, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_GLOBAL, 1, 1);

    error = cudaMemcpy(
        d_globalSeqDev, h_globalSeqDev, numSeqGlobal * sizeof(*d_globalSeqDev), cudaMemcpyHostToDevice
    );
    checkCudaError(error);
    error = cudaMemcpy(
        d_globalSeqIndexes, h_globalSeqIndexes, threadBlockCounter * sizeof(*d_globalSeqIndexes),
        cudaMemcpyHostToDevice
    );
    checkCudaError(error);

    quickSortGlobalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataTable, dataBuffer, d_globalSeqDev, d_globalSeqIndexes
    );

    error = cudaMemcpy(
        h_globalSeqDev, d_globalSeqDev, numSeqGlobal * sizeof(*h_globalSeqDev), cudaMemcpyDeviceToHost
    );
    checkCudaError(error);
}

/*
Finishes quicksort with local (one thread block processes one block) quicksort.
*/
void runQuickSortLocalKernel(
    data_t *dataTable, data_t *dataBuffer, loc_seq_t *h_localSeq, loc_seq_t *d_localSeq,
    uint_t numThreadBlocks, order_t sortOrder
)
{
    cudaError_t error;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block scan array size, array size for bitonic sort)
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL * sizeof(uint_t), THRESHOLD_BITONIC_SORT_LOCAL * sizeof(*dataTable)
    );
    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL, 1, 1);

    error = cudaMemcpy(d_localSeq, h_localSeq, numThreadBlocks * sizeof(*d_localSeq), cudaMemcpyHostToDevice);
    checkCudaError(error);

    if (sortOrder == ORDER_ASC)
    {
        quickSortLocalKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, dataBuffer, d_localSeq
        );
    }
    else
    {
        quickSortLocalKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, dataBuffer, d_localSeq
        );
    }
}

/*
Executes parallel quicksort.
*/
data_t* quickSort(
    data_t *h_dataInput, data_t *d_dataInput, data_t *d_dataBuffer, data_t *h_minMaxValues,
    h_glob_seq_t *h_globalSeqHost, h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev,
    d_glob_seq_t *d_globalSeqDev, uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes,
    loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t tableLen, order_t sortOrder
)
{
    // Because a lot of empty sequences can be generated, this counter is used to keep track of all
    // theoretically generated sequences.
    uint_t numSeqAll = 1;
    uint_t numSeqGlobal = 1; // Number of sequences for GLOBAL quicksort
    uint_t numSeqLocal = 0;  // Number of sequences for LOCAL quicksort
    uint_t numSeqLimit = (tableLen - 1) / THRESHOLD_PARTITION_SIZE_GLOBAL + 1;
    uint_t elemsPerThreadBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
    bool generateSequences = tableLen > THRESHOLD_PARTITION_SIZE_GLOBAL;
    data_t minVal, maxVal;

    // Searches for min and max value in input array
    minMaxReduction(
        h_dataInput, d_dataInput, (data_t*)d_dataBuffer, h_minMaxValues, tableLen, minVal, maxVal
    );
    // Null/zero distribution
    if (minVal == maxVal)
    {
        return d_dataInput;
    }
    h_globalSeqHost[0].setInitSeq(tableLen, minVal, maxVal);

    // GLOBAL QUICKSORT
    while (generateSequences)
    {
        uint_t threadBlockCounter = 0;

        // Transfers host sequences to device sequences (device needs different data about sequence than host)
        for (uint_t seqIdx = 0; seqIdx < numSeqGlobal; seqIdx++)
        {
            uint_t threadBlocksPerSeq = (h_globalSeqHost[seqIdx].length - 1) / elemsPerThreadBlock + 1;
            h_globalSeqDev[seqIdx].setFromHostSeq(h_globalSeqHost[seqIdx], threadBlockCounter, threadBlocksPerSeq);

            // For all thread blocks in current iteration marks, they are assigned to current sequence.
            for (uint_t blockIdx = 0; blockIdx < threadBlocksPerSeq; blockIdx++)
            {
                h_globalSeqIndexes[threadBlockCounter++] = seqIdx;
            }
        }

        runQuickSortGlobalKernel(
            d_dataInput, d_dataBuffer, h_globalSeqDev, d_globalSeqDev, h_globalSeqIndexes,
            d_globalSeqIndexes, numSeqGlobal, threadBlockCounter
        );

        uint_t numSeqGlobalOld = numSeqGlobal;
        numSeqGlobal = 0;
        numSeqAll *= 2;

        // Generates new sub-sequences and depending on their size adds them to list for GLOBAL or LOCAL quicksort
        // If theoretical number of sequences reached limit, sequences are transfered to array for LOCAL quicksort
        for (uint_t seqIdx = 0; seqIdx < numSeqGlobalOld; seqIdx++)
        {
            h_glob_seq_t seqHost = h_globalSeqHost[seqIdx];
            d_glob_seq_t seqDev = h_globalSeqDev[seqIdx];

            // New subsequece (lower)
            if (seqDev.offsetLower > THRESHOLD_PARTITION_SIZE_GLOBAL && numSeqAll < numSeqLimit)
            {
                h_globalSeqHostBuffer[numSeqGlobal++].setLowerSeq(seqHost, seqDev);
            }
            else if (seqDev.offsetLower > 0)
            {
                h_localSeq[numSeqLocal++].setLowerSeq(seqHost, seqDev);
            }

            // New subsequece (greater)
            if (seqDev.offsetGreater > THRESHOLD_PARTITION_SIZE_GLOBAL && numSeqAll < numSeqLimit)
            {
                h_globalSeqHostBuffer[numSeqGlobal++].setGreaterSeq(seqHost, seqDev);
            }
            else if (seqDev.offsetGreater > 0)
            {
                h_localSeq[numSeqLocal++].setGreaterSeq(seqHost, seqDev);
            }
        }

        h_glob_seq_t *temp = h_globalSeqHost;
        h_globalSeqHost = h_globalSeqHostBuffer;
        h_globalSeqHostBuffer = temp;

        generateSequences &= numSeqAll < numSeqLimit && numSeqGlobal > 0;
    }

    // If global quicksort was not used, than sequence is initialized for LOCAL quicksort
    if (tableLen <= THRESHOLD_PARTITION_SIZE_GLOBAL)
    {
        numSeqLocal++;
        h_localSeq[0].setInitSeq(tableLen);
    }

    runQuickSortLocalKernel(
        d_dataInput, d_dataBuffer, h_localSeq, d_localSeq, numSeqLocal, sortOrder
    );

    return d_dataBuffer;
}

/*
Sorts data wit parallel quicksort.
*/
double sortParallel(
    data_t *h_dataInput, data_t *h_dataOutput, data_t *d_dataTable, data_t *d_dataBuffer, data_t *h_minMaxValues,
    h_glob_seq_t *h_globalSeqHost, h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev,
    d_glob_seq_t *d_globalSeqDev, uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes,
    loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    data_t* d_dataResult = quickSort(
        h_dataInput, d_dataTable, d_dataBuffer, h_minMaxValues, h_globalSeqHost, h_globalSeqHostBuffer,
        h_globalSeqDev, d_globalSeqDev, h_globalSeqIndexes, d_globalSeqIndexes, h_localSeq, d_localSeq,
        tableLen, sortOrder
    );

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_dataOutput, d_dataResult, tableLen * sizeof(*h_dataOutput), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
