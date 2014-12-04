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
__device__ void compareExchange2(data_t *el1, data_t *el2)
{
    if ((*el1 > *el2) ^ sortOrder)
    {
        data_t temp = *el1;
        *el1 = *el2;
        *el2 = temp;
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

template <order_t sortOrder>
__device__ void compareExchange32(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8,
    data_t *el9, data_t *el10, data_t *el11, data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16,
    data_t *el17, data_t *el18, data_t *el19, data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24,
    data_t *el25, data_t *el26, data_t *el27, data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32
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
    compareExchange2<sortOrder>(el17, el18);
    compareExchange2<sortOrder>(el19, el20);
    compareExchange2<sortOrder>(el21, el22);
    compareExchange2<sortOrder>(el23, el24);
    compareExchange2<sortOrder>(el25, el26);
    compareExchange2<sortOrder>(el27, el28);
    compareExchange2<sortOrder>(el29, el30);
    compareExchange2<sortOrder>(el31, el32);

    compareExchange16<sortOrder>(
        el1, el17, el3, el19, el5, el21, el7, el23, el9, el25, el11, el27, el13, el29, el15, el31
    );
    compareExchange16<sortOrder>(
        el2, el18, el4, el20, el6, el22, el8, el24, el10, el26, el12, el28, el14, el30, el16, el32
    );
}

template <order_t sortOrder>
__device__ void load2(data_t *table, int_t stride, int_t tableLen, data_t *el1, data_t *el2)
{
    data_t val = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    *el1 = tableLen >= 0 ? table[0] : val;
    *el2 = stride <= tableLen ? table[stride] : val;
}

template <order_t sortOrder>
__device__ void store2(data_t *table, int_t stride, int_t tableLen, data_t el1, data_t el2)
{
    if (tableLen >= 0)
    {
        table[0] = el1;
    }
    if (stride <= tableLen)
    {
        table[stride] = el2;
    }
}

template <order_t sortOrder>
__device__ void load4(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4
)
{
    load2<sortOrder>(table, stride, tableLen, el1, el2);
    load2<sortOrder>(table + tableOffset, stride, tableLen - tableOffset, el3, el4);
}

template <order_t sortOrder>
__device__ void store4(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t el1, data_t el2, data_t el3,
    data_t el4
)
{
    store2<sortOrder>(table, stride, tableLen, el1, el2);
    store2<sortOrder>(table + tableOffset, stride, tableLen - tableOffset, el3, el4);
}

template <order_t sortOrder>
__device__ void load8(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8
)
{
    load4<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4);
    load4<sortOrder>(table + 2 * tableOffset, tableOffset, stride, tableLen - 2 * tableOffset, el5, el6, el7, el8);
}

template <order_t sortOrder>
__device__ void store8(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8
)
{
    store4<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4);
    store4<sortOrder>(table + 2 * tableOffset, tableOffset, stride, tableLen - 2 * tableOffset, el5, el6, el7, el8);
}

template <order_t sortOrder>
__device__ void load16(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11,
    data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16
)
{
    load8<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8);
    load8<sortOrder>(
        table + 4 * tableOffset, tableOffset, stride, tableLen - 4 * tableOffset, el9, el10, el11, el12, el13,
        el14, el15, el16
    );
}

template <order_t sortOrder>
__device__ void store16(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16
)
{
    store8<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8);
    store8<sortOrder>(
        table + 4 * tableOffset, tableOffset, stride, tableLen - 4 * tableOffset, el9, el10, el11, el12, el13,
        el14, el15, el16
    );
}

template <order_t sortOrder>
__device__ void load32(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11,
    data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16, data_t *el17, data_t *el18, data_t *el19,
    data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24, data_t *el25, data_t *el26, data_t *el27,
    data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32
    )
{
    load16<sortOrder>(
        table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16
    );
    load16<sortOrder>(
        table + 8 * tableOffset, tableOffset, stride, tableLen - 8 * tableOffset, el17, el18, el19, el20,
        el21, el22, el23, el24, el25, el26, el27, el28, el29, el30, el31, el32
    );
}

template <order_t sortOrder>
__device__ void store32(
    data_t *table, uint_t tableOffset, int_t stride, int_t tableLen, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16, data_t el17, data_t el18, data_t el19,
    data_t el20, data_t el21, data_t el22, data_t el23, data_t el24, data_t el25, data_t el26, data_t el27,
    data_t el28, data_t el29, data_t el30, data_t el31, data_t el32
    )
{
    store16<sortOrder>(
        table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16
    );
    store16<sortOrder>(
        table + 8 * tableOffset, tableOffset, stride, tableLen - 8 * tableOffset, el17, el18, el19, el20, el21,
        el22, el23, el24, el25, el26, el27, el28, el29, el30, el31, el32
    );
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
__global__ void multiStep1Kernel(data_t *table, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2;

    getMultiStepParams(step, 1, stride, tableOffset, indexTable);
    table += indexTable;
    tableLen -= indexTable + 1;

    load2<sortOrder>(table, stride, tableLen, &el1, &el2);
    compareExchange2<sortOrder>(&el1, &el2);
    store2<sortOrder>(table, stride, tableLen, el1, el2);
}

template __global__ void multiStep1Kernel<ORDER_ASC>(data_t *table, int_t tableLen, uint_t step);
template __global__ void multiStep1Kernel<ORDER_DESC>(data_t *table, int_t tableLen, uint_t step);


template <order_t sortOrder>
__global__ void multiStep2Kernel(data_t *table, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4;

    getMultiStepParams(step, 2, stride, tableOffset, indexTable);
    table += indexTable;
    tableLen -= indexTable + 1;

    load4<sortOrder>(table, tableOffset, stride, tableLen, &el1, &el2, &el3, &el4);
    compareExchange4<sortOrder>(&el1, &el2, &el3, &el4);
    store4<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4);
}

template __global__ void multiStep2Kernel<ORDER_ASC>(data_t *table, int_t tableLen, uint_t step);
template __global__ void multiStep2Kernel<ORDER_DESC>(data_t *table, int_t tableLen, uint_t step);


template <order_t sortOrder>
__global__ void multiStep3Kernel(data_t *table, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8;

    getMultiStepParams(step, 3, stride, tableOffset, indexTable);
    table += indexTable;
    tableLen -= indexTable + 1;

    load8<sortOrder>(table, tableOffset, stride, tableLen, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8);
    compareExchange8<sortOrder>(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8);
    store8<sortOrder>(table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8);
}

template __global__ void multiStep3Kernel<ORDER_ASC>(data_t *table, int_t tableLen, uint_t step);
template __global__ void multiStep3Kernel<ORDER_DESC>(data_t *table, int_t tableLen, uint_t step);


template <order_t sortOrder>
__global__ void multiStep4Kernel(data_t *table, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16;

    getMultiStepParams(step, 4, stride, tableOffset, indexTable);
    table += indexTable;
    tableLen -= indexTable + 1;

    load16<sortOrder>(
        table, tableOffset, stride, tableLen, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
    );
    compareExchange16<sortOrder>(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
    );
    store16<sortOrder>(
        table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7,
        el8, el9, el10, el11, el12, el13, el14, el15, el16
    );
}

template __global__ void multiStep4Kernel<ORDER_ASC>(data_t *table, int_t tableLen, uint_t step);
template __global__ void multiStep4Kernel<ORDER_DESC>(data_t *table, int_t tableLen, uint_t step);


template <order_t sortOrder>
__global__ void multiStep5Kernel(data_t *table, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16, el17,
        el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29, el30, el31, el32;

    getMultiStepParams(step, 5, stride, tableOffset, indexTable);
    table += indexTable;
    tableLen -= indexTable + 1;

    load32<sortOrder>(
        table, tableOffset, stride, tableLen, &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10,
        &el11, &el12, &el13, &el14, &el15, &el16, &el17, &el18, &el19, &el20, &el21, &el22, &el23, &el24,
        &el25, &el26, &el27, &el28, &el29, &el30, &el31, &el32
    );
    compareExchange32<sortOrder>(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16,
        &el17, &el18, &el19, &el20, &el21, &el22, &el23, &el24, &el25, &el26, &el27, &el28, &el29, &el30,
        &el31, &el32
    );
    store32<sortOrder>(
        table, tableOffset, stride, tableLen, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29,
        el30, el31, el32
    );
}

template __global__ void multiStep5Kernel<ORDER_ASC>(data_t *table, int_t tableLen, uint_t step);
template __global__ void multiStep5Kernel<ORDER_DESC>(data_t *table, int_t tableLen, uint_t step);


/*
Needed for first step of every phase.
*/
template <order_t sortOrder>
__global__ void bitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t step)
{
    uint_t stride = 1 << (step - 1);
    uint_t pairsPerThreadBlock = (THREADS_PER_GLOBAL_MERGE * ELEMS_PER_THREAD_GLOBAL_MERGE) >> 1;
    uint_t indexGlobal = blockIdx.x * pairsPerThreadBlock + threadIdx.x;

    for (uint_t i = 0; i < ELEMS_PER_THREAD_GLOBAL_MERGE >> 1; i++)
    {
        uint_t indexThread = indexGlobal + i * THREADS_PER_GLOBAL_MERGE;
        uint_t offset = ((indexThread & (stride - 1)) << 1) + 1;
        indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));

        uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
        if (index + offset >= tableLen)
        {
            break;
        }

        compareExchange2<sortOrder>(&dataTable[index], &dataTable[index + offset]);
    }
}

template __global__ void bitonicMergeGlobalKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeGlobalKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen, uint_t step);


/*
Global bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t step)
{
    extern __shared__ data_t mergeTile[];
    bool firstStepOfPhaseCopy = isFirstStepOfPhase;  // isFirstStepOfPhase is not editable (constant)

    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
    uint_t pairsPerBlockLength = dataBlockLength >> 1;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        mergeTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        for (uint_t tx = threadIdx.x; tx < pairsPerBlockLength; tx += THREADS_PER_LOCAL_MERGE)
        {
            uint_t indexThread = tx;
            uint_t offset = stride;

            // In normalized bitonic sort, first STEP of every PHASE uses different offset than all other STEPS.
            if (firstStepOfPhaseCopy)
            {
                offset = ((tx & (stride - 1)) << 1) + 1;
                indexThread = (tx / stride) * stride + ((stride - 1) - (tx % stride));
                firstStepOfPhaseCopy = false;
            }

            uint_t index = (indexThread << 1) - (indexThread & (stride - 1));
            if (index + offset >= dataBlockLength)
            {
                break;
            }

            compareExchange2<sortOrder>(&mergeTile[index], &mergeTile[index + offset]);
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        dataTable[offset + tx] = mergeTile[tx];
    }
}

template __global__ void bitonicMergeLocalKernel<ORDER_ASC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_ASC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
