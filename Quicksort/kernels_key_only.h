#ifndef KERNELS_KEY_ONLY_QUICKSORT_H
#define KERNELS_KEY_ONLY_QUICKSORT_H


#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "../Utils/constants_common.h"
#include "../Utils/kernels_utils.h"
#include "../BitonicSort/kernels_key_only_utils.h"
#include "data_types.h"
#include "constants.h"
#include "kernels_common_utils.h"


/*
Sorts input data with NORMALIZED bitonic sort (all comparisons are made in same direction,
easy to implement for input sequences of arbitrary size) and outputs them to output array.
*/
template <uint_t threadsBitonicSort, order_t sortOrder>
__device__ void normalizedBitonicSort(data_t *input, data_t *output, loc_seq_t localParams)
{
    extern __shared__ data_t bitonicSortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += threadsBitonicSort)
    {
        bitonicSortTile[tx] = input[localParams.start + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < localParams.length; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            if (stride == subBlockSize)
            {
                bitonicMergeStep<sortOrder, threadsBitonicSort, true>(
                    bitonicSortTile, 0, localParams.length, localParams.length, stride
                );
            }
            else
            {
                bitonicMergeStep<sortOrder, threadsBitonicSort, false>(
                    bitonicSortTile, 0, localParams.length, localParams.length, stride
                );
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += threadsBitonicSort)
    {
        output[localParams.start + tx] = bitonicSortTile[tx];
    }
}

/*
Executes global quicksort - multiple thread blocks process one sequence. They count how many elements are
lower/greater than pivot and then execute partitioning. At the end last thread block processing the
sequence stores the pivots.

TODO try alignment with 32 for coalasced reading
*/
template <uint_t threadsSortGlobal, uint_t elemsThreadGlobal, order_t sortOrder>
__global__ void quickSortGlobalKernel(
    data_t *dataInput, data_t *dataBuffer, d_glob_seq_t *sequences, uint_t *seqIndexes
)
{
    extern __shared__ data_t globalSortTile[];
#if USE_REDUCTION_IN_GLOBAL_SORT
    data_t *minValues = globalSortTile;
    data_t *maxValues = globalSortTile + threadsSortGlobal;
#endif

    // Index of sequence, which this thread block is partitioning
    __shared__ uint_t seqIdx;
    // Start and length of the data assigned to this thread block
    __shared__ uint_t localStart, localLength;
    __shared__ d_glob_seq_t sequence;

    if (threadIdx.x == (threadsSortGlobal - 1))
    {
        seqIdx = seqIndexes[blockIdx.x];
        sequence = sequences[seqIdx];
        uint_t elemsPerBlock = threadsSortGlobal * elemsThreadGlobal;
        uint_t localBlockIdx = blockIdx.x - sequence.startThreadBlockIdx;

        // Params.threadBlockCounter cannot be used, because it can get modified by other blocks.
        uint_t offset = localBlockIdx * elemsPerBlock;
        localStart = sequence.start + offset;
        localLength = offset + elemsPerBlock <= sequence.length ? elemsPerBlock : sequence.length - offset;
    }
    __syncthreads();

    data_t *primaryArray = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
    data_t *bufferArray = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataInput : dataBuffer;

#if USE_REDUCTION_IN_GLOBAL_SORT
    // Initializes min/max values.
    data_t minVal = MAX_VAL, maxVal = MIN_VAL;
#endif

    // Number of elements lower/greater than pivot (local for thread)
    uint_t localLower = 0, localGreater = 0;

    // Counts the number of elements lower/greater than pivot and finds min/max
    for (uint_t tx = threadIdx.x; tx < localLength; tx += threadsSortGlobal)
    {
        data_t temp = primaryArray[localStart + tx];
        localLower += temp < sequence.pivot;
        localGreater += temp > sequence.pivot;

#if USE_REDUCTION_IN_GLOBAL_SORT
        // Max value is calculated for "lower" sequence and min value is calculated for "greater" sequence.
        // Min for lower sequence and max of greater sequence (min and max of currently partitioned
        // sequence) were already calculated on host.
        maxVal = max(maxVal, temp < sequence.pivot ? temp : MIN_VAL);
        minVal = min(minVal, temp > sequence.pivot ? temp : MAX_VAL);
#endif
    }

#if USE_REDUCTION_IN_GLOBAL_SORT
    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Calculates and saves min/max values, before shared memory gets overriden by scan
    minMaxReduction<threadsSortGlobal>();
    if (threadIdx.x == (threadsSortGlobal - 1))
    {
        atomicMin(&sequences[seqIdx].greaterSeqMinVal, minValues[0]);
        atomicMax(&sequences[seqIdx].lowerSeqMaxVal, maxValues[0]);
    }
#endif
    __syncthreads();

    // Calculates number of elements lower/greater than pivot inside whole thread blocks
    uint_t scanLower = intraBlockScan<threadsSortGlobal>(localLower);
    __syncthreads();
    uint_t scanGreater = intraBlockScan<threadsSortGlobal>(localGreater);
    __syncthreads();

    // Calculates number of elements lower/greater than pivot for all thread blocks processing this sequence
    __shared__ uint_t globalLower, globalGreater;
    if (threadIdx.x == (threadsSortGlobal - 1))
    {
        globalLower = atomicAdd(&sequences[seqIdx].offsetLower, scanLower);
        globalGreater = atomicAdd(&sequences[seqIdx].offsetGreater, scanGreater);
    }
    __syncthreads();

    uint_t indexLower = sequence.start + globalLower + scanLower - localLower;
    uint_t indexGreater = sequence.start + sequence.length - globalGreater - scanGreater;

    // Scatters elements to newly generated left/right subsequences
    for (uint_t tx = threadIdx.x; tx < localLength; tx += threadsSortGlobal)
    {
        data_t temp = primaryArray[localStart + tx];

        if (temp < sequence.pivot)
        {
            bufferArray[indexLower++] = temp;
        }
        else if (temp > sequence.pivot)
        {
            bufferArray[indexGreater++] = temp;
        }
    }

    // Atomic sub has to be executed at the end of the kernel - after scattering of elements has been completed
    if (threadIdx.x == (threadsSortGlobal - 1))
    {
        sequence.threadBlockCounter = atomicSub(&sequences[seqIdx].threadBlockCounter, 1) - 1;
    }
    __syncthreads();

    // Last block assigned to current sub-sequence stores pivots
    if (sequence.threadBlockCounter == 0)
    {
        data_t pivot = sequence.pivot;

        uint_t index = sequence.start + sequences[seqIdx].offsetLower + threadIdx.x;
        uint_t end = sequence.start + sequence.length - sequences[seqIdx].offsetGreater;

        // Pivots have to be stored in output array, because they won't be moved anymore
        while (index < end)
        {
            dataBuffer[index] = pivot;
            index += threadsSortGlobal;
        }
    }
}

/*
Executes local quicksort - one thread block processes one sequence. It counts number of elements
lower/greater than pivot and then performs partitioning.
Workstack is used - shortest sequence is always processed.

TODO try alignment with 32 for coalasced reading
*/
template <uint_t threadsSortLocal, uint_t thresholdBitonicSort, order_t sortOrder>
__global__ void quickSortLocalKernel(data_t *dataInput, data_t *dataBuffer, loc_seq_t *sequences)
{
    // Explicit stack (instead of recursion), which holds sequences, which need to be processed.
    __shared__ loc_seq_t workstack[32];
    __shared__ int_t workstackCounter;

    // Global offset for scattering of pivots
    __shared__ uint_t pivotLowerOffset, pivotGreaterOffset;
    __shared__ data_t pivot;

    if (threadIdx.x == 0)
    {
        workstack[0] = sequences[blockIdx.x];
        workstackCounter = 0;
    }
    __syncthreads();

    while (workstackCounter >= 0)
    {
        __syncthreads();
        loc_seq_t sequence = popWorkstack(workstack, workstackCounter);

        if (sequence.length <= thresholdBitonicSort)
        {
            // Bitonic sort is executed in-place and sorted data has to be writter to output.
            data_t *inputTemp = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
            normalizedBitonicSort<threadsSortLocal, sortOrder>(
                inputTemp, dataBuffer, sequence
            );

            continue;
        }

        data_t *primaryArray = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
        data_t *bufferArray = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataInput : dataBuffer;

        if (threadIdx.x == 0)
        {
            pivot = getMedian(
                primaryArray[sequence.start], primaryArray[sequence.start + (sequence.length / 2)],
                primaryArray[sequence.start + sequence.length - 1]
            );
        }
        __syncthreads();

        // Counters for number of elements lower/greater than pivot
        uint_t localLower = 0, localGreater = 0;

        // Every thread counts the number of elements lower/greater than pivot
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += threadsSortLocal)
        {
            data_t temp = primaryArray[sequence.start + tx];
            localLower += temp < pivot;
            localGreater += temp > pivot;
        }

        // Calculates global offsets for each thread with inclusive scan
        uint_t globalLower = intraBlockScan<threadsSortLocal>(localLower);
        __syncthreads();
        uint_t globalGreater = intraBlockScan<threadsSortLocal>(localGreater);
        __syncthreads();

        uint_t indexLower = sequence.start + globalLower - localLower;
        uint_t indexGreater = sequence.start + sequence.length - globalGreater;

        // Scatter elements to newly generated left/right subsequences
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += threadsSortLocal)
        {
            data_t temp = primaryArray[sequence.start + tx];

            if (temp < pivot)
            {
                bufferArray[indexLower++] = temp;
            }
            else if (temp > pivot)
            {
                bufferArray[indexGreater++] = temp;
            }
        }

        // Pushes new subsequences on explicit stack and broadcast pivot offsets into shared memory
        if (threadIdx.x == (threadsSortLocal - 1))
        {
            pushWorkstack(workstack, workstackCounter, sequence, pivot, globalLower, globalGreater);

            pivotLowerOffset = globalLower;
            pivotGreaterOffset = globalGreater;
        }
        __syncthreads();

        // Scatters the pivots to output array. Pivots have to be stored in output array, because they
        // won't be moved anymore
        uint_t index = sequence.start + pivotLowerOffset + threadIdx.x;
        uint_t end = sequence.start + sequence.length - pivotGreaterOffset;

        while (index < end)
        {
            dataBuffer[index] = pivot;
            index += threadsSortLocal;
        }
    }
}

#endif
