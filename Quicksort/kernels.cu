#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


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
http://stackoverflow.com/questions/1582356/fastest-way-of-finding-the-middle-value-of-a-triple
*/
__device__ uint_t getMedian(uint_t a, uint_t b, uint_t c) {
    uint_t maxVal = max(max(a, b), c);
    uint_t minVal = min(min(a, b), c);

    return a ^ b ^ c ^ maxVal ^ minVal;
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

// TODO in general chech if __shared__ values work faster (pivot, array1, array2, ...)
// TODO try alignment with 32 because of bank conflicts.
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

    workstack[0] = localParams[blockIdx.x];
    workstackCounter = 1;

    while (workstackCounter > 0) {
        // TODO try with explicit local values start, end, direction
        lparam_t params = workstack[workstackCounter - 1];

        if (params.length <= BITONIC_SORT_SIZE_LOCAL) {
            // Bitonic sort is executed in-place and sorted data has to be writter to output.
            el_t *inputTemp = params.direction ? output : input;
            normalizedBitonicSort(inputTemp, output, params, tableLen, orderAsc);

            workstackCounter--;
            continue;
        }

        // In order not to spoil references *input and *output, additional 2 local references are used
        el_t *array1 = params.direction ? output : input;
        el_t *array2 = params.direction ? input : output;

        uint_t pivot = getMedian(
            array1[params.start].key, array1[(params.start + params.length) / 2].key,
            array1[params.start + params.length].key
        );

        uint_t lowerCounter = 0;
        uint_t greaterCounter = 0;
    }
}
