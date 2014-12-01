#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"


///////////////////////////////////////////////////////////////////
////////////////////////////// UTILS //////////////////////////////
///////////////////////////////////////////////////////////////////

/*
Compares 2 elements and exchanges them according to orderAsc.
*/
template <order_t sortOrder>
__device__ void compareExchange2(data_t *elem1, data_t *elem2)
{
    if ((*elem1 > *elem2) ^ sortOrder)
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

template <order_t sortOrder>
__device__ void compareExchange4(data_t *el1, data_t *el2, data_t *el3, data_t *el4)
{
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);

    compareExchange2<sortOrder>(el1, el3);
    compareExchange2<sortOrder>(el2, el4);
}

template <order_t sortOrder>
__device__ void compareExchange8(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8
)
{
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);
    compareExchange2<sortOrder>(el5, el6);
    compareExchange2<sortOrder>(el7, el8);

    compareExchange4<sortOrder>(el1, el5, el3, el7);
    compareExchange4<sortOrder>(el2, el6, el4, el8);
}

template <order_t sortOrder>
__device__ void compareExchange16(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8,
    data_t *el9, data_t *el10, data_t *el11, data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16
)
{
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);
    compareExchange2<sortOrder>(el5, el6);
    compareExchange2<sortOrder>(el7, el8);
    compareExchange2<sortOrder>(el9, el10);
    compareExchange2<sortOrder>(el11, el12);
    compareExchange2<sortOrder>(el13, el14);
    compareExchange2<sortOrder>(el15, el16);

    compareExchange8<sortOrder>(el1, el9, el3, el11, el5, el13, el7, el15);
    compareExchange8<sortOrder>(el2, el10, el4, el12, el6, el14, el8, el16);
}

__device__ void load2(data_t *table, uint_t stride, data_t *el1, data_t *el2)
{
    *el1 = table[0];
    *el2 = table[stride];
}

__device__ void store2(data_t *table, uint_t stride, data_t el1, data_t el2)
{
    table[0] = el1;
    table[stride] = el2;
}

__device__ void load4(
    data_t *table, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3, data_t *el4
)
{
    load2(table, stride, el1, el2);
    load2(table + tableOffset, stride, el3, el4);
}

__device__ void store4(
    data_t *table, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3, data_t el4
)
{
    store2(table, stride, el1, el2);
    store2(table + tableOffset, stride, el3, el4);
}

__device__ void load8(
    data_t *table, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3, data_t *el4,
    data_t *el5, data_t *el6, data_t *el7, data_t *el8
)
{
    load4(table, tableOffset, stride, el1, el2, el3, el4);
    load4(table + 2 * tableOffset, tableOffset, stride, el5, el6, el7, el8);
}

__device__ void store8(
    data_t *table, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3, data_t el4,
    data_t el5, data_t el6, data_t el7, data_t el8
)
{
    store4(table, tableOffset, stride, el1, el2, el3, el4);
    store4(table + 2 * tableOffset, tableOffset, stride, el5, el6, el7, el8);
}

__device__ void load16(
    data_t *table, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3, data_t *el4,
    data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11, data_t *el12,
    data_t *el13, data_t *el14, data_t *el15, data_t *el16
)
{
    load8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    load8(table + 4 * tableOffset, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16);
}

__device__ void store16(
    data_t *table, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3, data_t el4,
    data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11, data_t el12,
    data_t el13, data_t el14, data_t el15, data_t el16
)
{
    store8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    store8(table + 4 * tableOffset, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16);
}

/*
Generates parameters needed for multistep.
> stride - (gap) between two elements beeing compared
> threadsPerSubBlocks - how many threads apper per sub-block in current step
> indexTable - start index, at which thread should start fetching elements
*/
__device__ void getMultiStepParams(
    uint_t step, uint_t degree, uint_t &stride, uint_t &threadsPerSubBlock, uint_t &indexTable
)
{
    uint_t indexThread = blockIdx.x * blockDim.x + threadIdx.x;

    stride = 1 << (step - 1);
    threadsPerSubBlock = 1 << (step - degree);
    indexTable = (indexThread >> (step - degree) << step) + indexThread % threadsPerSubBlock;
}


///////////////////////////////////////////////////////////////////
///////////////////////////// KERNELS /////////////////////////////
///////////////////////////////////////////////////////////////////

/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        bitonicSortTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (uint_t subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (uint_t stride = subBlockSize; stride > 0; stride >>= 1)
        {
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT)
            {
                uint_t indexThread = tx;
                uint_t offset = stride;

                // In NORMALIZED bitonic sort, first STEP of every PHASE uses different offset than all other
                // STEPS. Also, in first step of every phase, offset sizes are generated in ASCENDING order
                // (normalized bitnic sort requires DESCENDING order). Because of that, we can break the loop if
                // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
                // than thread indexes inside every sub-block have to be reversed.
                if (stride == subBlockSize)
                {
                    indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                    offset = ((tx & (stride - 1)) << 1) + 1;
                }

                uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
                if (index + offset >= dataBlockLength)
                {
                    break;
                }

                compareExchange2<sortOrder>(&bitonicSortTile[index], &bitonicSortTile[index + offset]);
            }

            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen);


template <order_t sortOrder>
__global__ void multiStep1Kernel(data_t *table, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    bool direction;
    data_t el1, el2;

    getMultiStepParams(step, 1, stride, tableOffset, indexTable);
    table += indexTable;

    load2(table, stride, &el1, &el2);
    compareExchange2<sortOrder>(&el1, &el2);
    store2(table, stride, el1, el2);
}

template __global__ void multiStep1Kernel<ORDER_ASC>(data_t *table, uint_t step);
template __global__ void multiStep1Kernel<ORDER_DESC>(data_t *table, uint_t step);


//__global__ void multiStep2Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
//    uint_t stride, tableOffset, indexTable;
//    bool direction;
//    el_t el1, el2, el3, el4;
//
//    getMultiStepParams(phase, step, 2, stride, tableOffset, indexTable, direction);
//    table += indexTable;
//
//    load4(table, tableOffset, stride, &el1, &el2, &el3, &el4);
//    compareExchange4(&el1, &el2, &el3, &el4, direction ^ orderAsc);
//    store4(table, tableOffset, stride, el1, el2, el3, el4);
//}
//
//__global__ void multiStep3Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
//    uint_t stride, tableOffset, indexTable;
//    bool direction;
//    el_t el1, el2, el3, el4, el5, el6, el7, el8;
//
//    getMultiStepParams(phase, step, 3, stride, tableOffset, indexTable, direction);
//    table += indexTable;
//
//    load8(table, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8);
//    compareExchange8(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, direction ^ orderAsc);
//    store8(table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
//}
//
//__global__ void multiStep4Kernel(el_t *table, uint_t phase, uint_t step, bool orderAsc) {
//    uint_t stride, tableOffset, indexTable;
//    bool direction;
//    el_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16;
//
//    getMultiStepParams(phase, step, 4, stride, tableOffset, indexTable, direction);
//    table += indexTable;
//
//    load16(
//        table, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
//        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
//    );
//    compareExchange16(
//        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13,
//        &el14, &el15, &el16, direction ^ orderAsc
//    );
//    store16(
//        table, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7,
//        el8, el9, el10, el11, el12, el13, el14, el15, el16
//    );
//}
//
///*
//Global bitonic merge for sections, where stride IS GREATER OR EQUAL than max shared memory.
//*/
//__global__ void bitonicMergeKernel(el_t *table, uint_t phase, bool orderAsc) {
//    extern __shared__ el_t mergeTile[];
//    uint_t index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
//    // Elements inside same sub-block have to be ordered in same direction
//    bool direction = orderAsc ^ ((index >> phase) & 1);
//
//    // Every thread loads 2 elements
//    mergeTile[threadIdx.x] = table[index];
//    mergeTile[blockDim.x + threadIdx.x] = table[blockDim.x + index];
//
//    // Bitonic merge
//    for (uint_t stride = blockDim.x; stride > 0; stride >>= 1) {
//        __syncthreads();
//        uint_t start = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
//        compareExchange2(&mergeTile[start], &mergeTile[start + stride], direction);
//    }
//
//    __syncthreads();
//    table[index] = mergeTile[threadIdx.x];
//    table[blockDim.x + index] = mergeTile[blockDim.x + threadIdx.x];
//}
