#include <stdio.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__global__ void printElemsKernel(el_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i].key);
    }
    printf("\n");
}

__global__ void printDataKernel(data_t *table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        printf("%2d ", table[i]);
    }
    printf("\n");
}

/*
Compares 2 elements and exchanges them according to orderAsc.
*/
template <typename T>
__device__ void compareExchange(T *elem1, T *elem2, order_t sortOrder) {
    if (((int_t)(elem1->key - elem2->key) > 0) ^ sortOrder) {
        T temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

template <typename T>
__device__ void bitonicSort(T *dataTable, uint_t tableLen, order_t sortOrder) {
    extern __shared__ T bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Read data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1) {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1) {
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT) {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
                if (stride == subBlockSize) {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength) {
                    break;
                }

                compareExchange(&bitonicSortTile[index], &bitonicSortTile[index + offset], sortOrder);
            }
            __syncthreads();
        }
    }

    // Store data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort and collects samples in array for local samples.
*/
template <typename T>
__global__ void bitonicSortCollectSamplesKernel(T *dataTable, data_t *localSamples, uint_t tableLen,
                                                order_t sortOrder) {
    extern __shared__ T bitonicSortTile[];

    bitonicSort<T>(dataTable, tableLen, sortOrder);

    // After sort has been performed, samples are scattered to array of local samples
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t localSamplesDistance = elemsPerThreadBlock / NUM_SAMPLES;
    uint_t samplesPerThreadBlock = (elemsPerThreadBlock - 1) / localSamplesDistance + 1;

    for (uint_t tx = threadIdx.x; tx < samplesPerThreadBlock; tx += THREADS_PER_BITONIC_SORT) {
        localSamples[blockIdx.x * NUM_SAMPLES + tx] = bitonicSortTile[tx * localSamplesDistance].key;
    }
}

template __global__ void bitonicSortCollectSamplesKernel<el_t>(
    el_t *dataTable, data_t *localSamples, uint_t tableLen, order_t sortOrder
);

///*
//Global bitonic merge for sections, where stride IS GREATER than max shared memory.
//*/
//template <typename T>
//__global__ void bitonicMergeGlobalKernel(T *dataTable, uint_t tableLen, uint_t step, bool firstStepOfPhase,
//                                         order_t sortOrder) {
//    uint_t stride = 1 << (step - 1);
//    uint_t pairsPerThreadBlock = (THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE) >> 1;
//    uint_t indexGlobal = blockIdx.x * pairsPerThreadBlock + threadIdx.x;
//
//    for (uint_t i = 0; i < ELEMS_PER_THREAD_GLOBAL_MERGE >> 1; i++) {
//        uint_t indexThread = indexGlobal + i * THREADS_PER_GLOBAL_MERGE;
//        uint_t offset = stride;
//
//        // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
//        if (firstStepOfPhase) {
//            offset = ((indexThread & (stride - 1)) << 1) + 1;
//            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
//        }
//
//        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
//        if (index + offset >= tableLen) {
//            break;
//        }
//
//        compareExchange(&dataTable[index], &dataTable[index + offset], sortOrder);
//    }
//}
//
///*
//Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
//*/
//__global__ void bitonicMergeLocalKernel(el_t *dataTable, uint_t tableLen, uint_t step, bool isFirstStepOfPhase,
//    order_t sortOrder) {
//    extern __shared__ el_t mergeTile[];
//
//    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
//    uint_t offset = blockIdx.x * elemsPerThreadBlock;
//    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
//    uint_t pairsPerBlockLength = dataBlockLength >> 1;
//
//    // Read data from global to shared memory.
//    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
//        mergeTile[tx] = dataTable[offset + tx];
//    }
//    __syncthreads();
//
//    // Bitonic merge
//    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1) {
//        for (uint_t tx = threadIdx.x; tx < pairsPerBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
//            uint_t indexThread = tx;
//            uint_t offset = stride;
//
//            // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
//            if (isFirstStepOfPhase) {
//                offset = ((tx & (stride - 1)) << 1) + 1;
//                indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
//                isFirstStepOfPhase = false;
//            }
//
//            uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
//            if (index + offset >= dataBlockLength) {
//                break;
//            }
//
//            compareExchange(&mergeTile[index], &mergeTile[index + offset], sortOrder);
//        }
//        __syncthreads();
//    }
//
//    // Stores data from shared to global memory
//    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE) {
//        dataTable[offset + tx] = mergeTile[tx];
//    }
//}
