#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"

///////////////////////////////////////////////////////////////////
////////////////////////////// UTILS //////////////////////////////
///////////////////////////////////////////////////////////////////


////////////////////////// GENERAL UTILS //////////////////////////

/*
http://stackoverflow.com/questions/1582356/fastest-way-of-finding-the-middle-value-of-a-triple
*/
__device__ uint_t getMedian(uint_t a, uint_t b, uint_t c) {
    uint_t maxVal = max(max(a, b), c);
    uint_t minVal = min(min(a, b), c);

    return a ^ b ^ c ^ maxVal ^ minVal;
}


/////////////////////////// SCAN UTILS ////////////////////////////

/*
Performs scan and computes, how many elements have 'true' predicate before current element.
*/
__device__ uint_t intraWarpScan(volatile uint_t *scanTile, uint_t val, uint_t stride) {
    // The same kind of indexing as for bitonic sort
    uint_t index = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

    scanTile[index] = 0;
    index += stride;
    scanTile[index] = val;

    if (stride > 1) {
        scanTile[index] += scanTile[index - 1];
    }
    if (stride > 2) {
        scanTile[index] += scanTile[index - 2];
    }
    if (stride > 4) {
        scanTile[index] += scanTile[index - 4];
    }
    if (stride > 8) {
        scanTile[index] += scanTile[index - 8];
    }
    if (stride > 16) {
        scanTile[index] += scanTile[index - 16];
    }

    // Converts inclusive scan to exclusive
    return scanTile[index] - val;
}

__device__ uint_t intraBlockScan(uint_t val) {
    extern __shared__ uint_t scanTile[];
    // If thread block size is lower than warp size, than thread block size is used as warp size
    uint_t warpLen = warpSize <= blockDim.x ? warpSize : blockDim.x;
    uint_t warpIdx = threadIdx.x / warpLen;
    uint_t laneIdx = threadIdx.x & (warpLen - 1);  // Thread index inside warp

    uint_t warpResult = intraWarpScan(scanTile, val, warpLen);
    __syncthreads();

    if (laneIdx == warpLen - 1) {
        scanTile[warpIdx] = warpResult;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < blockDim.x / warpLen) {
        scanTile[threadIdx.x] = intraWarpScan(scanTile, scanTile[threadIdx.x], blockDim.x / warpLen);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx];
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
__device__ void normalizedBitonicSort(el_t *input, el_t *output, lparam_t localParams, uint_t tableLen, bool orderAsc) {
    extern __shared__ el_t sortTile[];

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += blockDim.x) {
        sortTile[tx] = input[localParams.start + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < localParams.length; subBlockSize <<= 1) {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            for (uint_t tx = threadIdx.x; tx < (tableLen / MAX_SEQUENCES) >> 1; tx += blockDim.x) {
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

                compareExchange(&sortTile[index], &sortTile[index + offset], orderAsc);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < localParams.length; tx += blockDim.x) {
        output[localParams.start + tx] = sortTile[tx];
    }
}


////////////////////// LOCAL QUICKSORT UTILS //////////////////////

__device__ void pushNewSeqOnStack(lparam_t *workstack, lparam_t params, uint_t &workstackCounter,
                                  uint_t lowerCounter, uint_t greaterCounter) {
    lparam_t newParams1, newParams2;

    newParams1.direction = !params.direction;
    newParams2.direction = !params.direction;

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
    workstackCounter--;
    if (newParams1.length > 0) {
        workstack[++workstackCounter] = newParams1;
    }
    if (newParams1.length > 0) {
        workstack[++workstackCounter] = newParams2;
    }
}


///////////////////////////////////////////////////////////////////
///////////////////////////// KERNELS /////////////////////////////
///////////////////////////////////////////////////////////////////

// TODO in general chech if __shared__ values work faster (pivot, array1, array2, ...)
// TODO try alignment with 32 because of bank conflicts in shared memory.
__global__ void quickSortLocalKernel(el_t *input, el_t *output, lparam_t *localParams, uint_t tableLen,
                                     bool orderAsc) {
    __shared__ extern uint_t localSortTile[];

    // Array of counters for elements lower/greater than pivot. One element belongs to one thread.
    uint_t *lowerThanPivot = localSortTile;
    uint_t *greaterThanPivot = localSortTile + blockDim.x;

    // Explicit stack (instead of recursion) for work to be done
    // TODO allocate memory dynamically according to sub-block size
    __shared__ lparam_t workstack[32];
    __shared__ uint_t workstackCounter;

    __shared__ uint_t pivotLowerOffset;
    __shared__ uint_t pivotGreaterOffset;

    if (threadIdx.x == 0) {
        workstack[0] = localParams[blockIdx.x];
        workstackCounter = 0;
    }
    __syncthreads();

    // TODO handle this on host
    if (workstack[0].length == 0) {
        return;
    }

    while (workstackCounter >= 0) {
        // TODO try with explicit local values start, end, direction
        lparam_t params = workstack[workstackCounter];

        if (params.length <= BITONIC_SORT_SIZE_LOCAL) {
            // Bitonic sort is executed in-place and sorted data has to be writter to output.
            el_t *inputTemp = params.direction ? output : input;

            normalizedBitonicSort(inputTemp, output, params, tableLen, orderAsc);
            // TODO verify if syncthreads() is needed
            __syncthreads();

            workstackCounter--;
            continue;
        }

        // In order not to spoil references *input and *output, additional 2 local references are used
        el_t *array1 = params.direction ? output : input;
        el_t *array2 = params.direction ? input : output;

        uint_t pivot = getMedian(
            array1[params.start].key, array1[params.start + (params.length / 2)].key,
            array1[params.start + params.length - 1].key
        );

        // Counter of number of elements, which are lower/greater than pivot
        uint_t localLower = 0;
        uint_t localGreater = 0;

        // Every thread counts the number of elements lower/greater than pivot
        for (uint_t tx = threadIdx.x; tx < params.length; tx += blockDim.x) {
            el_t temp = array1[params.start + tx];
            localLower += temp.key < pivot;
            localGreater += temp.key > pivot;
        }
        __syncthreads();

        // Calculates global offsets for each thread
        uint_t globalLower = intraBlockScan(localLower);
        __syncthreads();
        uint_t globalGreater = intraBlockScan(localGreater);
        __syncthreads();

        // Add new subsequences on explicit stack
        // TODO verify for thread block size is greater than table len
        // TODO move bellow scattering
        if (threadIdx.x == (blockDim.x - 1)) {
            pushNewSeqOnStack(
                workstack, params, workstackCounter, globalLower + localLower, globalGreater + localGreater
            );

            pivotLowerOffset = globalLower;
            pivotGreaterOffset = globalGreater;
        }
        __syncthreads();

        uint_t indexLower = params.start + globalLower;
        uint_t indexGreater = params.start + params.length - (globalGreater + localGreater);

        // Scatter elements to newly generated left/right subsequences
        for (uint_t tx = threadIdx.x; tx < params.length; tx += blockDim.x) {
            el_t temp = array1[params.start + tx];

            if (temp.key < pivot) {
                array2[indexLower++] = temp;
            } else if (temp.key > pivot) {
                array2[indexGreater++] = temp;
            }
        }
        __syncthreads();

        // Scatter pivots
        /*for (uint_t index = params.start + ;;) {

        }*/

        workstackCounter--;
    }
}
