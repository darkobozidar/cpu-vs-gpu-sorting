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
#include "kernels_key_value.h"
#include "sort.h"


/*
Runs global (multiple thread blocks process one sequence) quicksort and coppies required data to and
from device.
*/
void QuicksortParallel::runQuickSortGlobalKernel(
    data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, data_t *d_valuesPivot,
    d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev, uint_t *h_globalSeqIndexes,
    uint_t *d_globalSeqIndexes, uint_t numSeqGlobal, uint_t threadBlockCounter
)
{
    cudaError_t error;

    // 1. arg: Size of array for calculation of min/max value ("2" because of MIN and MAX)
    // 2. arg: Size of array needed to perform scan of counters for number of elements lower/greater than
    //         pivot ("2" because of intra-warp scan).
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_GLOBAL_KV * sizeof(data_t), 2 * THREADS_PER_SORT_GLOBAL_KV * sizeof(uint_t)
    );
    dim3 dimGrid(threadBlockCounter, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_GLOBAL_KV, 1, 1);

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
        d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_valuesPivot, d_globalSeqDev,
        d_globalSeqIndexes
    );

    error = cudaMemcpy(
        h_globalSeqDev, d_globalSeqDev, numSeqGlobal * sizeof(*h_globalSeqDev), cudaMemcpyDeviceToHost
    );
    checkCudaError(error);
}

/*
Finishes quicksort with local (one thread block processes one block) quicksort.
*/
template <order_t sortOrder>
void QuicksortParallel::runQuickSortLocalKernel(
    data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, data_t *d_valuesPivot,
    loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t numThreadBlocks
)
{
    cudaError_t error;

    // The same shared memory array is used for counting elements greater/lower than pivot and for bitonic sort.
    // max(intra-block scan array size, array size for bitonic sort ("2 *" because of key-value pairs))
    uint_t sharedMemSize = max(
        2 * THREADS_PER_SORT_LOCAL_KV * sizeof(uint_t), 2 * THRESHOLD_BITONIC_SORT_LOCAL_KV * sizeof(*d_keys)
    );
    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(THREADS_PER_SORT_LOCAL_KV, 1, 1);

    error = cudaMemcpy(d_localSeq, h_localSeq, numThreadBlocks * sizeof(*d_localSeq), cudaMemcpyHostToDevice);
    checkCudaError(error);

    quickSortLocalKernel<sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_valuesPivot, d_localSeq
    );
}

/*
Executes parallel quicksort.
*/
template <order_t sortOrder>
void QuicksortParallel::quicksortParallel(
    data_t *h_keys, data_t *&d_keys, data_t *&d_values, data_t *&d_keysBuffer, data_t *&d_valuesBuffer,
    data_t *d_valuesPivot, data_t *h_minMaxValues, h_glob_seq_t *h_globalSeqHost,
    h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev,
    uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, loc_seq_t *h_localSeq, loc_seq_t *d_localSeq,
    uint_t arrayLength
)
{
    // Because a lot of empty sequences can be generated, this counter is used to keep track of all
    // theoretically generated sequences.
    uint_t numSeqAll = 1;
    uint_t numSeqGlobal = 1; // Number of sequences for GLOBAL quicksort
    uint_t numSeqLocal = 0;  // Number of sequences for LOCAL quicksort
    uint_t numSeqLimit = (arrayLength - 1) / THRESHOLD_PARTITION_SIZE_GLOBAL_KV + 1;
    uint_t elemsPerThreadBlock = THREADS_PER_SORT_GLOBAL_KV * ELEMENTS_PER_THREAD_GLOBAL_KV;
    bool generateSequences = arrayLength > THRESHOLD_PARTITION_SIZE_GLOBAL_KV;
    data_t minVal, maxVal;

    // Searches for min and max value in input array
    minMaxReduction<THRESHOLD_REDUCTION_KV, THREADS_PER_REDUCTION_KV, ELEMENTS_PER_THREAD_REDUCTION_KV>(
        h_keys, d_keys, d_keysBuffer, h_minMaxValues, arrayLength, minVal, maxVal
    );
    // Null/zero distribution
    if (minVal == maxVal)
    {
        data_t *temp = d_keys;
        d_keys = d_keysBuffer;
        d_keysBuffer = temp;

        temp = d_values;
        d_values = d_valuesBuffer;
        d_valuesBuffer = temp;
        return;
    }
    h_globalSeqHost[0].setInitSeq(arrayLength, minVal, maxVal);

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
            d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_valuesPivot, h_globalSeqDev, d_globalSeqDev,
            h_globalSeqIndexes, d_globalSeqIndexes, numSeqGlobal, threadBlockCounter
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
            if (seqDev.offsetLower > THRESHOLD_PARTITION_SIZE_GLOBAL_KV && numSeqAll < numSeqLimit)
            {
                h_globalSeqHostBuffer[numSeqGlobal++].setLowerSeq(seqHost, seqDev);
            }
            else if (seqDev.offsetLower > 0)
            {
                h_localSeq[numSeqLocal++].setLowerSeq(seqHost, seqDev);
            }

            // New subsequece (greater)
            if (seqDev.offsetGreater > THRESHOLD_PARTITION_SIZE_GLOBAL_KV && numSeqAll < numSeqLimit)
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
    if (arrayLength <= THRESHOLD_PARTITION_SIZE_GLOBAL_KV)
    {
        numSeqLocal++;
        h_localSeq[0].setInitSeq(arrayLength);
    }

    runQuickSortLocalKernel<sortOrder>(
        d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_valuesPivot, h_localSeq, d_localSeq, numSeqLocal
    );
}

/*
Wrapper for bitonic sort method.
The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
*/
void QuicksortParallel::sortKeyValue()
{
    if (_sortOrder == ORDER_ASC)
    {
        quicksortParallel<ORDER_ASC>(
            _h_keys, _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_valuesPivot, _h_minMaxValues,
            _h_globalSeqHost, _h_globalSeqHostBuffer, _h_globalSeqDev, _d_globalSeqDev, _h_globalSeqIndexes,
            _d_globalSeqIndexes, _h_localSeq, _d_localSeq, _arrayLength
        );
    }
    else
    {
        quicksortParallel<ORDER_DESC>(
            _h_keys, _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_valuesPivot, _h_minMaxValues,
            _h_globalSeqHost, _h_globalSeqHostBuffer, _h_globalSeqDev, _d_globalSeqDev, _h_globalSeqIndexes,
            _d_globalSeqIndexes, _h_localSeq, _d_localSeq, _arrayLength
        );
    }
}
