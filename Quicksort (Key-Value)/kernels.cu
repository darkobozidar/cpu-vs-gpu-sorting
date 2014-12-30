#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"
#include "data_types.h"


//////////////////////////// GENERAL UTILS //////////////////////////

/*
Calculates the next power of 2 of provided value or returns value if it is already a power of 2.
*/
__device__ uint_t nextPowerOf2Device(uint_t value)
{
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;

    return value;
}


///////////////////////////// SCAN UTILS ////////////////////////////

/*
Performs exclusive scan and computes, how many elements have 'true' predicate before current element.
*/
template <uint_t blockSize>
__device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val)
{
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

    scanTile[index] = 0;
    index += min(blockSize, WARP_SIZE);
    scanTile[index] = val;

    if (blockSize >= 2)
    {
        scanTile[index] += scanTile[index - 1];
    }
    if (blockSize >= 4)
    {
        scanTile[index] += scanTile[index - 2];
    }
    if (blockSize >= 8)
    {
        scanTile[index] += scanTile[index - 4];
    }
    if (blockSize >= 16)
    {
        scanTile[index] += scanTile[index - 8];
    }
    if (blockSize >= 32)
    {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

/*
Performs intra-block INCLUSIVE scan.
*/
template <uint_t blockSize>
__device__ uint_t intraBlockScan(uint_t val)
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / WARP_SIZE;
    uint_t laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

    uint_t warpResult = intraWarpScan<blockSize>(scanTile, val);
    __syncthreads();

    if (laneIdx == WARP_SIZE - 1)
    {
        scanTile[warpIdx] = warpResult + val;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < blockSize / WARP_SIZE)
    {
        scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx] + val;
}


////////////////////////// MIN/MAX REDUCTION ////////////////////////

/*
Performs parallel min/max reduction. Half of the threads in thread block calculates min value,
other half calculates max value. Result is returned as the first element in each array.
*/
__device__ void minMaxReduction(uint_t length)
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + THREADS_PER_REDUCTION;

    for (uint_t stride = length / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            minValues[threadIdx.x] = min(minValues[threadIdx.x], minValues[threadIdx.x + stride]);
        }
        else if (threadIdx.x < 2 * stride)
        {
            maxValues[threadIdx.x - stride] = max(maxValues[threadIdx.x - stride], maxValues[threadIdx.x]);
        }
        __syncthreads();
    }
}

/*
Min reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
__device__ void warpMinReduce(volatile data_t *minValues)
{
    uint_t index = (threadIdx.x >> WARP_SIZE_LOG << (WARP_SIZE_LOG + 1)) + (threadIdx.x & (WARP_SIZE - 1));

    if (blockSize >= 64)
    {
        minValues[index] = min(minValues[index], minValues[index + 32]);
    }
    if (blockSize >= 32)
    {
        minValues[index] = min(minValues[index], minValues[index + 16]);
    }
    if (blockSize >= 16)
    {
        minValues[index] = min(minValues[index], minValues[index + 8]);
    }
    if (blockSize >= 8)
    {
        minValues[index] = min(minValues[index], minValues[index + 4]);
    }
    if (blockSize >= 4)
    {
        minValues[index] = min(minValues[index], minValues[index + 2]);
    }
    if (blockSize >= 2)
    {
        minValues[index] = min(minValues[index], minValues[index + 1]);
    }
}

/*
Max reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
__device__ void warpMaxReduce(volatile data_t *maxValues) {
    uint_t tx = threadIdx.x - blockSize / 2;
    uint_t index = (tx >> WARP_SIZE_LOG << (WARP_SIZE_LOG + 1)) + (tx & (WARP_SIZE - 1));

    if (blockSize >= 64)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 32]);
    }
    if (blockSize >= 32)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 16]);
    }
    if (blockSize >= 16)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 8]);
    }
    if (blockSize >= 8)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 4]);
    }
    if (blockSize >= 4)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 2]);
    }
    if (blockSize >= 2)
    {
        maxValues[index] = max(maxValues[index], maxValues[index + 1]);
    }
}


///////////////////////// BITONIC SORT UTILS ////////////////////////

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
__device__ void compareExchange(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Sorts input data with NORMALIZED bitonic sort (all comparisons are made in same direction,
easy to implement for input sequences of arbitrary size) and outputs them to output array.
*/
template <order_t sortOrder>
__device__ void normalizedBitonicSort(data_t *input, data_t *output, loc_seq_t localParams)
{
    extern __shared__ data_t bitonicSortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += THREADS_PER_SORT_LOCAL)
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
            for (uint_t tx = threadIdx.x; tx < localParams.length >> 1; tx += THREADS_PER_SORT_LOCAL)
            {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other
                // STEPS. Also in first step of every phase, offsets sizes are generated in ASCENDING order
                // (normalized bitnic sort requires DESCENDING order). Because of that we can break the loop if
                // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
                // than thread indexes inside every sub-block have to be reversed.
                if (stride == subBlockSize)
                {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= localParams.length)
                {
                    break;
                }

                compareExchange<sortOrder>(&bitonicSortTile[index], &bitonicSortTile[index + offset]);
            }

            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += THREADS_PER_SORT_LOCAL)
    {
        output[localParams.start + tx] = bitonicSortTile[tx];
    }
}


//////////////////////// LOCAL QUICKSORT UTILS //////////////////////

/*
Returns last local sequence on workstack and decreases workstack counter (pop).
*/
__device__ loc_seq_t popWorkstack(loc_seq_t *workstack, int_t &workstackCounter)
{
    if (threadIdx.x == 0)
    {
        workstackCounter--;
    }
    __syncthreads();

    return workstack[workstackCounter + 1];
}

/*
From provided sequence generates 2 new sequences and pushes them on stack of sequences.
*/
__device__ int_t pushWorkstack(
    loc_seq_t *workstack, int_t &workstackCounter, loc_seq_t sequence, data_t pivot, uint_t lowerCounter,
    uint_t greaterCounter
)
{
    loc_seq_t newSequence1, newSequence2;

    newSequence1.direction = (direct_t)!sequence.direction;
    newSequence2.direction = (direct_t)!sequence.direction;
    bool isLowerShorter = lowerCounter <= greaterCounter;

    // From provided sequence generates new sequences
    newSequence1.start = isLowerShorter ? sequence.start + sequence.length - greaterCounter : sequence.start;
    newSequence1.length = isLowerShorter ? greaterCounter : lowerCounter;
    newSequence1.minVal = isLowerShorter ? pivot : sequence.minVal;
    newSequence1.maxVal = isLowerShorter ? sequence.maxVal : pivot;

    newSequence2.start = isLowerShorter ? sequence.start : sequence.start + sequence.length - greaterCounter;
    newSequence2.length = isLowerShorter ? lowerCounter : greaterCounter;
    newSequence2.minVal = isLowerShorter ? sequence.minVal : pivot;
    newSequence2.maxVal = isLowerShorter ? pivot : sequence.maxVal;

    // Push news sequences on stack
    if (newSequence1.length > 0)
    {
        workstack[++workstackCounter] = newSequence1;
    }
    if (newSequence2.length > 0)
    {
        workstack[++workstackCounter] = newSequence2;
    }

    return workstackCounter;
}


/////////////////////////////////////////////////////////////////////
/////////////////////////////// KERNELS /////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
From input array finds min/max value and outputs the min/max value to output.
*/
__global__ void minMaxReductionKernel(data_t *input, data_t *output, uint_t tableLen)
{
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + THREADS_PER_REDUCTION;

    uint_t elemsPerBlock = THREADS_PER_REDUCTION * ELEMENTS_PER_THREAD_REDUCTION;
    uint_t offset = blockIdx.x * elemsPerBlock;
    uint_t dataBlockLength = offset + elemsPerBlock <= tableLen ? elemsPerBlock : tableLen - offset;

    data_t minVal = MAX_VAL;
    data_t maxVal = MIN_VAL;

    // Every thread reads and processes multiple elements
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_REDUCTION)
    {
        data_t val = input[offset + tx];
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
    }

    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Once all threads have processed their corresponding elements, than reduction is done in shared memory
    if (threadIdx.x < THREADS_PER_REDUCTION / 2)
    {
        warpMinReduce<THREADS_PER_REDUCTION>(minValues);
    }
    else
    {
        warpMaxReduce<THREADS_PER_REDUCTION>(maxValues);
    }
    __syncthreads();

    // First warp loads results from all othwer warps and performs reduction
    if ((threadIdx.x >> WARP_SIZE_LOG) == 0)
    {
        // Every warp reduces 2 * warpSize elements
        uint_t index = threadIdx.x << (WARP_SIZE_LOG + 1);

        // Threads load results of all other warp and half of those warps performs reduction on results
        if (index < THREADS_PER_REDUCTION && THREADS_PER_REDUCTION > WARP_SIZE)
        {
            minValues[threadIdx.x] = minValues[index];
            maxValues[threadIdx.x] = maxValues[index];

            if (index < THREADS_PER_REDUCTION / 2)
            {
                warpMinReduce<(THREADS_PER_REDUCTION >> (WARP_SIZE_LOG + 1))>(minValues);
            }
            else
            {
                warpMaxReduce<(THREADS_PER_REDUCTION >> (WARP_SIZE_LOG + 1))>(maxValues);
            }
        }

        // First thread in thread block outputs reduced results
        if (threadIdx.x == 0)
        {
            output[blockIdx.x] = minValues[0];
            output[gridDim.x + blockIdx.x] = maxValues[0];
        }
    }
}

/*
Executes global quicksort - multiple thread blocks process one sequence. They count how many elements are
lower/greater than pivot and then execute partitioning. At the end last thread block processing the
sequence stores the pivots.

TODO try alignment with 32 for coalasced reading
*/
__global__ void quickSortGlobalKernel(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, data_t *pivotValues,
    d_glob_seq_t *sequences, uint_t *seqIndexes
)
{
    extern __shared__ data_t globalSortTile[];
    data_t *minValues = globalSortTile;
    data_t *maxValues = globalSortTile + THREADS_PER_SORT_GLOBAL;

    uint_t elemsPerBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
    // Index of sequence, which this thread block is partitioning
    __shared__ uint_t seqIdx;
    // Start and length of the data assigned to this thread block
    __shared__ uint_t localStart, localLength, numActiveThreads;
    __shared__ d_glob_seq_t sequence;

    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1))
    {
        seqIdx = seqIndexes[blockIdx.x];
        sequence = sequences[seqIdx];
        uint_t localBlockIdx = blockIdx.x - sequence.startThreadBlockIdx;

        // Params.threadBlockCounter cannot be used, because it can get modified by other blocks.
        uint_t offset = localBlockIdx * elemsPerBlock;
        localStart = sequence.start + offset;
        localLength = offset + elemsPerBlock <= sequence.length ? elemsPerBlock : sequence.length - offset;
        numActiveThreads = nextPowerOf2Device(min(THREADS_PER_SORT_GLOBAL, localLength));
    }
    __syncthreads();

    data_t *keysPrimary = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataKeys : bufferKeys;
    data_t *valuesPrimary = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataValues : bufferValues;
    data_t *keysBuffer = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataKeys : bufferKeys;
    data_t *valuesBuffer = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataValues : bufferValues;

#if USE_REDUCTION_IN_GLOBAL_SORT
    // Initializes min/max values.
    data_t minVal = MAX_VAL, maxVal = MIN_VAL;
#endif

    // Number of elements lower/greater than pivot (local for thread)
    uint_t localLower = 0, localGreater = 0;

    // Counts the number of elements lower/greater than pivot and finds min/max
    for (uint_t tx = threadIdx.x; tx < localLength; tx += THREADS_PER_SORT_GLOBAL)
    {
        data_t temp = keysPrimary[localStart + tx];
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
    minMaxReduction(numActiveThreads);
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1))
    {
        atomicMin(&sequences[seqIdx].greaterSeqMinVal, minValues[0]);
        atomicMax(&sequences[seqIdx].lowerSeqMaxVal, maxValues[0]);
    }
#endif
    __syncthreads();

    // Calculates number of elements lower/greater than pivot inside whole thread blocks
    uint_t scanLower = intraBlockScan<THREADS_PER_SORT_GLOBAL>(localLower);
    __syncthreads();
    uint_t scanGreater = intraBlockScan<THREADS_PER_SORT_GLOBAL>(localGreater);
    __syncthreads();

    // Calculates number of elements lower/greater than pivot for all thread blocks processing this sequence
    __shared__ uint_t globalLower, globalGreater;
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1))
    {
        globalLower = atomicAdd(&sequences[seqIdx].offsetLower, scanLower);
        globalGreater = atomicAdd(&sequences[seqIdx].offsetGreater, scanGreater);
    }
    __syncthreads();

    uint_t indexLower = sequence.start + globalLower + scanLower - localLower;
    uint_t indexGreater = sequence.start + sequence.length - globalGreater - scanGreater;
    uint_t indexPivot = sequence.start + (sequence.length - globalLower - globalGreater);
    indexPivot += elemsPerBlock - scanLower - scanGreater - localLower - localGreater;

    // Scatters elements to newly generated left/right subsequences
    for (uint_t tx = threadIdx.x; tx < localLength; tx += THREADS_PER_SORT_GLOBAL)
    {
        data_t key = keysPrimary[localStart + tx];
        data_t value = valuesPrimary[localStart + tx];

        if (key < sequence.pivot)
        {
            keysBuffer[indexLower] = key;
            valuesBuffer[indexLower] = value;
            indexLower++;
        }
        else if (key > sequence.pivot)
        {
            keysBuffer[indexGreater] = key;
            valuesBuffer[indexGreater] = value;
            indexGreater++;
        }
        else
        {
            pivotValues[indexPivot++] = value;
        }
    }

    // Atomic sub has to be executed at the end of the kernel - after scattering of elements has been completed
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1))
    {
        sequence.threadBlockCounter = atomicSub(&sequences[seqIdx].threadBlockCounter, 1) - 1;
    }
    __syncthreads();

    // Last block assigned to current sub-sequence stores pivots
    if (sequence.threadBlockCounter == 0)
    {
        data_t pivot = sequence.pivot;
        uint_t globalOffsetLower = sequences[seqIdx].offsetLower;
        uint_t globalOffsetGreater = sequences[seqIdx].offsetGreater;

        uint_t indexOutput = sequence.start + globalOffsetLower + threadIdx.x;
        uint_t endOutput = sequence.start + sequence.length - globalOffsetGreater;
        uint_t indexPivot = sequence.start;

        // Pivots have to be stored in output array, because they won't be moved anymore
        while (indexOutput < endOutput)
        {
            bufferKeys[indexOutput] = pivot;
            bufferValues[indexOutput] = pivotValues[indexPivot];

            indexOutput += THREADS_PER_SORT_GLOBAL;
            indexPivot += THREADS_PER_SORT_GLOBAL;
        }
    }
}

/*
Executes local quicksort - one thread block processes one sequence. It counts number of elements
lower/greater than pivot and then performs partitioning.
Workstack is used - shortest sequence is always processed.

TODO try alignment with 32 for coalasced reading
*/
template <order_t sortOrder>
__global__ void quickSortLocalKernel(data_t *dataInput, data_t *dataBuffer, loc_seq_t *sequences)
{
    // Explicit stack (instead of recursion), which holds sequences, which need to be processed.
    __shared__ loc_seq_t workstack[32];
    __shared__ int_t workstackCounter;

    // Global offset for scattering of pivots
    __shared__ uint_t pivotLowerOffset, pivotGreaterOffset;

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
        data_t pivot = (sequence.minVal + sequence.maxVal) / 2;

        if (sequence.length <= THRESHOLD_BITONIC_SORT_LOCAL)
        {
            // Bitonic sort is executed in-place and sorted data has to be writter to output.
            data_t *inputTemp = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
            normalizedBitonicSort<sortOrder>(inputTemp, dataBuffer, sequence);

            continue;
        }

        data_t *primaryArray = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
        data_t *bufferArray = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataInput : dataBuffer;

        // Counters for number of elements lower/greater than pivot
        uint_t localLower = 0, localGreater = 0;

        // Every thread counts the number of elements lower/greater than pivot
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += THREADS_PER_SORT_LOCAL)
        {
            data_t temp = primaryArray[sequence.start + tx];
            localLower += temp < pivot;
            localGreater += temp > pivot;
        }

        // Calculates global offsets for each thread with inclusive scan
        uint_t globalLower = intraBlockScan<THREADS_PER_SORT_LOCAL>(localLower);
        __syncthreads();
        uint_t globalGreater = intraBlockScan<THREADS_PER_SORT_LOCAL>(localGreater);
        __syncthreads();

        uint_t indexLower = sequence.start + globalLower - localLower;
        uint_t indexGreater = sequence.start + sequence.length - globalGreater;

        // Scatter elements to newly generated left/right subsequences
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += THREADS_PER_SORT_LOCAL)
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
        if (threadIdx.x == (THREADS_PER_SORT_LOCAL - 1))
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
            index += THREADS_PER_SORT_LOCAL;
        }
    }
}

template __global__ void quickSortLocalKernel<ORDER_ASC>(
    data_t *dataInput, data_t *dataBuffer, loc_seq_t *sequences
);
template __global__ void quickSortLocalKernel<ORDER_DESC>(
    data_t *dataInput, data_t *dataBuffer, loc_seq_t *sequences
);
