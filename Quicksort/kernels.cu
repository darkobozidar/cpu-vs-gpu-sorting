#include <stdio.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"

///////////////////////////////////////////////////////////////////
////////////////////////////// UTILS //////////////////////////////
///////////////////////////////////////////////////////////////////


__global__ void printTableKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i].key);
    }
    printf("\n");
}

////////////////////////// GENERAL UTILS //////////////////////////

/*
http://stackoverflow.com/questions/1582356/fastest-way-of-finding-the-middle-value-of-a-triple
*/
__device__ data_t getMedian(data_t a, data_t b, data_t c) {
    data_t maxVal = max(max(a, b), c);
    data_t minVal = min(min(a, b), c);
    return a ^ b ^ c ^ maxVal ^ minVal;
}

__device__ uint_t nextPowerOf2(uint_t value) {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;

    return value;
}


/////////////////////////// SCAN UTILS ////////////////////////////

/*
Performs scan and computes, how many elements have 'true' predicate before current element.
*/
template <uint_t blockSize>
__device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val) {
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

    scanTile[index] = 0;
    index += min(blockSize, WARP_SIZE);
    scanTile[index] = val;

    if (blockSize >= 2) {
        scanTile[index] += scanTile[index - 1];
    }
    if (blockSize >= 4) {
        scanTile[index] += scanTile[index - 2];
    }
    if (blockSize >= 8) {
        scanTile[index] += scanTile[index - 4];
    }
    if (blockSize >= 16) {
        scanTile[index] += scanTile[index - 8];
    }
    if (blockSize >= 32) {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

/*
Performs intra-block INCLUSIVE scan.
*/
template <uint_t blockSize>
__device__ uint_t intraBlockScan(uint_t val) {
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / WARP_SIZE;
    uint_t laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

    uint_t warpResult = intraWarpScan<blockSize>(scanTile, val);
    __syncthreads();

    if (laneIdx == WARP_SIZE - 1) {
        scanTile[warpIdx] = warpResult + val;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < blockSize / WARP_SIZE) {
        scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx] + val;
}


//////////////////////// MIN/MAX REDUCTION ////////////////////////

/*
Performs parallel min/max reduction. Half of the threads in thread block calculates min value,
other half calculates max value. Result is returned as the first element in each array.
*/
__device__ void minMaxReduction(uint_t length) {
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + THREADS_PER_REDUCTION;

    for (uint_t stride = length / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            minValues[threadIdx.x] = min(minValues[threadIdx.x], minValues[threadIdx.x + stride]);
        } else if (threadIdx.x < 2 * stride) {
            maxValues[threadIdx.x - stride] = max(maxValues[threadIdx.x - stride], maxValues[threadIdx.x]);
        }
        __syncthreads();
    }
}

/*
Min reduction for warp. Every warp can reduce 64 elements or less.
*/
template <uint_t blockSize>
__device__ void warpMinReduce(volatile data_t *minValues) {
    uint_t index = (threadIdx.x >> WARP_SIZE_LOG << (WARP_SIZE_LOG + 1)) + (threadIdx.x & (WARP_SIZE - 1));

    if (blockSize >= 64) {
        minValues[index] = min(minValues[index], minValues[index + 32]);
    }
    if (blockSize >= 32) {
        minValues[index] = min(minValues[index], minValues[index + 16]);
    }
    if (blockSize >= 16) {
        minValues[index] = min(minValues[index], minValues[index + 8]);
    }
    if (blockSize >= 8) {
        minValues[index] = min(minValues[index], minValues[index + 4]);
    }
    if (blockSize >= 4) {
        minValues[index] = min(minValues[index], minValues[index + 2]);
    }
    if (blockSize >= 2) {
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

    if (blockSize >= 64) {
        maxValues[index] = max(maxValues[index], maxValues[index + 32]);
    }
    if (blockSize >= 32) {
        maxValues[index] = max(maxValues[index], maxValues[index + 16]);
    }
    if (blockSize >= 16) {
        maxValues[index] = max(maxValues[index], maxValues[index + 8]);
    }
    if (blockSize >= 8) {
        maxValues[index] = max(maxValues[index], maxValues[index + 4]);
    }
    if (blockSize >= 4) {
        maxValues[index] = max(maxValues[index], maxValues[index + 2]);
    }
    if (blockSize >= 2) {
        maxValues[index] = max(maxValues[index], maxValues[index + 1]);
    }
}


/////////////////////// BITONIC SORT UTILS ////////////////////////

/*
Compares 2 elements and exchanges them according to orderAsc.
*/
__device__ void compareExchange(el_t *elem1, el_t *elem2, bool orderAsc) {
    if (((int_t)(elem1->key - elem2->key) <= 0) ^ orderAsc) {
        el_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Sorts input data with NORMALIZED bitonic sort (all comparisons are made in same direction,
easy to implement for input sequences of arbitrary size) and outputs them to output array.

- TODO use quick sort kernel instead of bitonic sort
*/
__device__ void normalizedBitonicSort(el_t *input, el_t *output, loc_seq_t localParams, bool orderAsc) {
    extern __shared__ el_t bitonicSortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += THREADS_PER_SORT_LOCAL) {
        bitonicSortTile[tx] = input[localParams.start + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < localParams.length; subBlockSize <<= 1) {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            for (uint_t tx = threadIdx.x; tx < localParams.length >> 1; tx += THREADS_PER_SORT_LOCAL) {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other
                // STEPS. Also in first step of every phase, offsets sizes are generated in ASCENDING order
                // (normalized bitnic sort requires DESCENDING order). Because of that we can break the loop if
                // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
                // than thread indexes inside every sub-block have to be reversed.
                if (stride == subBlockSize) {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= localParams.length) {
                    break;
                }

                compareExchange(&bitonicSortTile[index], &bitonicSortTile[index + offset], orderAsc);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += THREADS_PER_SORT_LOCAL) {
        output[localParams.start + tx] = bitonicSortTile[tx];
    }
}


////////////////////// LOCAL QUICKSORT UTILS //////////////////////

__device__ loc_seq_t popWorkstack(loc_seq_t *workstack, int_t &workstackCounter) {
    if (threadIdx.x == 0) {
        workstackCounter--;
    }
    __syncthreads();

    return workstack[workstackCounter + 1];
}

__device__ int_t pushWorkstack(loc_seq_t *workstack, int_t &workstackCounter, loc_seq_t params,
                               uint_t lowerCounter, uint_t greaterCounter) {
    loc_seq_t newParams1, newParams2;

    newParams1.direction = (direct_t)!params.direction;
    newParams2.direction = (direct_t)!params.direction;

    // TODO try in-place change directly on workstack without newParams 1 and 2 - if not possible move to struct construcor
    if (lowerCounter <= greaterCounter) {
        newParams1.start = params.start + params.length - greaterCounter;
        newParams1.length = greaterCounter;
        newParams2.start = params.start;
        newParams2.length = lowerCounter;
    } else {
        newParams1.start = params.start;
        newParams1.length = lowerCounter;
        newParams2.start = params.start + params.length - greaterCounter;
        newParams2.length = greaterCounter;
    }

    // TODO verify if there are any benefits with this if statement
    if (newParams1.length > 0) {
        workstack[++workstackCounter] = newParams1;
    }
    if (newParams2.length > 0) {
        workstack[++workstackCounter] = newParams2;
    }

    return workstackCounter;
}


///////////////////////////////////////////////////////////////////
///////////////////////////// KERNELS /////////////////////////////
///////////////////////////////////////////////////////////////////

/*
From input array finds min/max value and outputs the min/max value to output.
*/
__global__ void minMaxReductionKernel(el_t *input, data_t *output, uint_t tableLen) {
    extern __shared__ data_t reductionTile[];
    data_t *minValues = reductionTile;
    data_t *maxValues = reductionTile + THREADS_PER_REDUCTION;

    uint_t elemsPerBlock = THREADS_PER_REDUCTION * ELEMENTS_PER_THREAD_REDUCTION;
    uint_t offset = blockIdx.x * elemsPerBlock;
    uint_t dataBlockLength = offset + elemsPerBlock <= tableLen ? elemsPerBlock : tableLen - offset;

    data_t minVal = MAX_VAL;
    data_t maxVal = MIN_VAL;

    // Every thread reads and processes multiple elements
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_REDUCTION) {
        data_t val = input[offset + tx].key;
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
    }

    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Once all threads have processed their corresponding elements, than reduction is done in shared memory
    if (threadIdx.x < THREADS_PER_REDUCTION / 2) {
        warpMinReduce<THREADS_PER_REDUCTION>(minValues);
    } else {
        warpMaxReduce<THREADS_PER_REDUCTION>(maxValues);
    }
    __syncthreads();

    // First warp loads results from all othwer warps and performs reduction
    if ((threadIdx.x >> WARP_SIZE_LOG) == 0) {
        // Every warp reduces 2 * warpSize elements
        uint_t index = threadIdx.x << (WARP_SIZE_LOG + 1);

        // Threads load results of all other warp and half of those warps performs reduction on results
        if (index < THREADS_PER_REDUCTION && THREADS_PER_REDUCTION > WARP_SIZE) {
            minValues[threadIdx.x] = minValues[index];
            maxValues[threadIdx.x] = maxValues[index];

            if (index < THREADS_PER_REDUCTION / 2) {
                warpMinReduce<(THREADS_PER_REDUCTION >> (WARP_SIZE_LOG + 1))>(minValues);
            } else {
                warpMaxReduce<(THREADS_PER_REDUCTION >> (WARP_SIZE_LOG + 1))>(maxValues);
            }
        }

        if (threadIdx.x == 0) {
            output[blockIdx.x] = minValues[0];
            output[gridDim.x + blockIdx.x] = maxValues[0];
        }
    }
}

// TODO try alignment with 32 for coalasced reading
__global__ void quickSortGlobalKernel(el_t *dataInput, el_t *dataBuffer, d_glob_seq_t *sequences, uint_t *seqIndexes) {
    extern __shared__ uint_t globalSortTile[];
    data_t *minValues = globalSortTile;
    data_t *maxValues = globalSortTile + THREADS_PER_SORT_GLOBAL;

    // Index of sequence, which this thread block is partitioning
    __shared__ uint_t seqIdx;
    // Start and length of the data assigned to this thread block
    __shared__ uint_t localStart, localLength, numActiveThreads;
    __shared__ d_glob_seq_t sequence;

    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1)) {
        seqIdx = seqIndexes[blockIdx.x];
        sequence = sequences[seqIdx];
        uint_t elemsPerBlock = THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL;
        uint_t localBlockIdx = blockIdx.x - sequence.startThreadBlockIdx;

        // Params.threadBlockCounter cannot be used, because it can get modified by other blocks.
        uint_t offset = localBlockIdx * elemsPerBlock;
        localStart = sequence.start + offset;
        localLength = offset + elemsPerBlock <= sequence.length ? elemsPerBlock : sequence.length - offset;
        numActiveThreads = nextPowerOf2(min(THREADS_PER_SORT_GLOBAL, localLength));
    }
    __syncthreads();

    el_t *primaryArray = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
    el_t *bufferArray = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataInput : dataBuffer;

#if USE_REDUCTION_IN_GLOBAL_SORT
    // Initializes min/max values.
    data_t minVal = MAX_VAL, maxVal = MIN_VAL;
#endif

    // Number of elements lower/greater than pivot (local for thread)
    uint_t localLower = 0, localGreater = 0;

    // Counts the number of elements lower/greater than pivot and finds min/max
    for (uint_t tx = threadIdx.x; tx < localLength; tx += THREADS_PER_SORT_GLOBAL) {
        el_t temp = primaryArray[localStart + tx];
        localLower += temp.key < sequence.pivot;
        localGreater += temp.key > sequence.pivot;

#if USE_REDUCTION_IN_GLOBAL_SORT
        // Max value is calculated for "lower" sequence and min value is calculated for "greater" sequence.
        // Min for lower sequence and max of greater sequence (min and max of currently partitioned
        // sequence) were already calculated on host.
        maxVal = max(maxVal, temp.key < sequence.pivot ? temp.key : MIN_VAL);
        minVal = min(minVal, temp.key > sequence.pivot ? temp.key : MAX_VAL);
#endif
    }

#if USE_REDUCTION_IN_GLOBAL_SORT
    minValues[threadIdx.x] = minVal;
    maxValues[threadIdx.x] = maxVal;
    __syncthreads();

    // Calculates and saves min/max values, before shared memory gets overriden by scan
    minMaxReduction(numActiveThreads);
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1)) {
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
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1)) {
        globalLower = atomicAdd(&sequences[seqIdx].offsetLower, scanLower);
        globalGreater = atomicAdd(&sequences[seqIdx].offsetGreater, scanGreater);
    }
    __syncthreads();

    uint_t indexLower = sequence.start + globalLower + scanLower - localLower;
    uint_t indexGreater = sequence.start + sequence.length - globalGreater - scanGreater;

    // Scatters elements to newly generated left/right subsequences
    for (uint_t tx = threadIdx.x; tx < localLength; tx += THREADS_PER_SORT_GLOBAL) {
        el_t temp = primaryArray[localStart + tx];

        if (temp.key < sequence.pivot) {
            bufferArray[indexLower++] = temp;
        } else if (temp.key > sequence.pivot) {
            bufferArray[indexGreater++] = temp;
        }
    }

    // Atomic sub has to be executed at the end of the kernel - after scattering of elements has been completed
    if (threadIdx.x == (THREADS_PER_SORT_GLOBAL - 1)) {
        sequence.threadBlockCounter = atomicSub(&sequences[seqIdx].threadBlockCounter, 1) - 1;
    }
    __syncthreads();

    // Last block assigned to current sub-sequence stores pivots
    if (sequence.threadBlockCounter == 0) {
        el_t pivot;
        pivot.key = sequence.pivot;

        uint_t index = sequence.start + sequences[seqIdx].offsetLower + threadIdx.x;
        uint_t end = sequence.start + sequence.length - sequences[seqIdx].offsetGreater;

        // Pivots have to be stored in output array, because they won't be moved anymore
        while (index < end) {
            dataBuffer[index] = pivot;
            index += THREADS_PER_SORT_GLOBAL;
        }
    }
}

// TODO in general chech if __shared__ values work faster (pivot, array1, array2, ...)
// TODO try alignment with 32 for coalasced reading
__global__ void quickSortLocalKernel(el_t *dataInput, el_t *dataBuffer, loc_seq_t *sequences, bool orderAsc) {
    // Explicit stack (instead of recursion), which holds sequences, which need to be processed.
    // TODO allocate explicit stack dynamically according to sub-block size
    __shared__ loc_seq_t workstack[32];
    __shared__ int_t workstackCounter;

    // Global offset for scattering of pivots
    __shared__ uint_t pivotLowerOffset, pivotGreaterOffset;
    __shared__ data_t pivot;

    if (threadIdx.x == 0) {
        workstack[0] = sequences[blockIdx.x];
        workstackCounter = 0;
    }
    __syncthreads();

    while (workstackCounter >= 0) {
        __syncthreads();
        loc_seq_t sequence = popWorkstack(workstack, workstackCounter);

        if (sequence.length <= THRESHOLD_BITONIC_SORT_LOCAL) {
            // Bitonic sort is executed in-place and sorted data has to be writter to output.
            el_t *inputTemp = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
            normalizedBitonicSort(inputTemp, dataBuffer, sequence, orderAsc);

            continue;
        }

        el_t *primaryArray = sequence.direction == PRIMARY_MEM_TO_BUFFER ? dataInput : dataBuffer;
        el_t *bufferArray = sequence.direction == BUFFER_TO_PRIMARY_MEM ? dataInput : dataBuffer;

        if (threadIdx.x == 0) {
            pivot = getMedian(
                primaryArray[sequence.start].key, primaryArray[sequence.start + (sequence.length / 2)].key,
                primaryArray[sequence.start + sequence.length - 1].key
            );
        }
        __syncthreads();

        // Counters for number of elements lower/greater than pivot
        uint_t localLower = 0, localGreater = 0;

        // Every thread counts the number of elements lower/greater than pivot
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += THREADS_PER_SORT_LOCAL) {
            el_t temp = primaryArray[sequence.start + tx];
            localLower += temp.key < pivot;
            localGreater += temp.key > pivot;
        }

        // Calculates global offsets for each thread with inclusive scan
        uint_t globalLower = intraBlockScan<THREADS_PER_SORT_LOCAL>(localLower);
        __syncthreads();
        uint_t globalGreater = intraBlockScan<THREADS_PER_SORT_LOCAL>(localGreater);
        __syncthreads();

        uint_t indexLower = sequence.start + globalLower - localLower;
        uint_t indexGreater = sequence.start + sequence.length - globalGreater;

        // Scatter elements to newly generated left/right subsequences
        for (uint_t tx = threadIdx.x; tx < sequence.length; tx += THREADS_PER_SORT_LOCAL) {
            el_t temp = primaryArray[sequence.start + tx];

            if (temp.key < pivot) {
                bufferArray[indexLower++] = temp;
            } else if (temp.key > pivot) {
                bufferArray[indexGreater++] = temp;
            }
        }

        // Pushes new subsequences on explicit stack and broadcast pivot offsets into shared memory
        if (threadIdx.x == (THREADS_PER_SORT_LOCAL - 1)) {
            pushWorkstack(workstack, workstackCounter, sequence, globalLower, globalGreater);

            pivotLowerOffset = globalLower;
            pivotGreaterOffset = globalGreater;
        }
        __syncthreads();

        // Scatters the pivots to output array. Pivots have to be stored in output array, because they won't be moved anymore
        el_t pivotEl;
        pivotEl.key = pivot;

        uint_t index = sequence.start + pivotLowerOffset + threadIdx.x;
        uint_t end = sequence.start + sequence.length - pivotGreaterOffset;

        while (index < end) {
            dataBuffer[index] = pivotEl;
            index += THREADS_PER_SORT_LOCAL;
        }
    }
}
