#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../Utils/data_types_common.h"
#include "constants.h"
#include "kernels_common_utils.h"
#include "kernels_key_only_utils.h"


/*
Sorts sub-blocks of input data with NORMALIZED bitonic sort.
*/
template <order_t sortOrder>
__global__ void bitonicSortKernel(data_t *dataTable, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT_KO * ELEMS_PER_THREAD_BITONIC_SORT_KO;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT_KO)
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
            for (uint_t tx = threadIdx.x; tx < dataBlockLength >> 1; tx += THREADS_PER_BITONIC_SORT_KO)
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
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT_KO)
    {
        dataTable[offset + tx] = bitonicSortTile[tx];
    }
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *dataTable, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *dataTable, uint_t tableLen);


/*
Performs bitonic merge with 1-multistep (sorts 2 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep1Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2;

    getMultiStepParams(step, 1, stride, tableOffset, indexTable);

    load2<sortOrder>(table + indexTable, table + tableLen, stride, &el1, &el2);
    compareExchange2<sortOrder>(&el1, &el2);
    store2(table + indexTable, table + tableLen, stride, el1, el2);
}

template __global__ void multiStep1Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep1Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);

/*
Performs bitonic merge with 2-multistep (sorts 4 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep2Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4;

    getMultiStepParams(step, 2, stride, tableOffset, indexTable);

    load4<sortOrder>(table + indexTable, table + tableLen, tableOffset, stride, &el1, &el2, &el3, &el4);
    compareExchange4<sortOrder>(&el1, &el2, &el3, &el4);
    store4(table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4);
}

template __global__ void multiStep2Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep2Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);


/*
Performs bitonic merge with 3-multistep (sorts 8 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep3Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8;

    getMultiStepParams(step, 3, stride, tableOffset, indexTable);

    load8<sortOrder>(
        table + indexTable, table + tableLen, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6,
        &el7, &el8
    );
    compareExchange8<sortOrder>(&el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8);
    store8(
        table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8
    );
}

template __global__ void multiStep3Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep3Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);


/*
Performs bitonic merge with 4-multistep (sorts 16 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep4Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16;

    getMultiStepParams(step, 4, stride, tableOffset, indexTable);

    load16<sortOrder>(
        table + indexTable, table + tableLen, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
    );
    compareExchange16<sortOrder>(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16
    );
    store16(
        table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8,
        el9, el10, el11, el12, el13, el14, el15, el16
    );
}

template __global__ void multiStep4Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep4Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);


/*
Performs bitonic merge with 5-multistep (sorts 32 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep5Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16, el17,
        el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29, el30, el31, el32;

    getMultiStepParams(step, 5, stride, tableOffset, indexTable);

    load32<sortOrder>(
        table + indexTable, table + tableLen, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16, &el17, &el18, &el19, &el20, &el21,
        &el22, &el23, &el24, &el25, &el26, &el27, &el28, &el29, &el30, &el31, &el32
    );
    compareExchange32<sortOrder>(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16,
        &el17, &el18, &el19, &el20, &el21, &el22, &el23, &el24, &el25, &el26, &el27, &el28, &el29, &el30,
        &el31, &el32
    );
    store32(
        table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8,
        el9, el10, el11, el12, el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25,
        el26, el27, el28, el29, el30, el31, el32
    );
}

template __global__ void multiStep5Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep5Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);


/*
Performs bitonic merge with 6-multistep (sorts 64 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep6Kernel(data_t *table, uint_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15, el16, el17,
        el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29, el30, el31, el32, el33,
        el34, el35, el36, el37, el38, el39, el40, el41, el42, el43, el44, el45, el46, el47, el48, el49,
        el50, el51, el52, el53, el54, el55, el56, el57, el58, el59, el60, el61, el62, el63, el64;

    getMultiStepParams(step, 6, stride, tableOffset, indexTable);

    load64<sortOrder>(
        table + indexTable, table + tableLen, tableOffset, stride, &el1, &el2, &el3, &el4, &el5, &el6, &el7,
        &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16, &el17, &el18, &el19, &el20, &el21,
        &el22, &el23, &el24, &el25, &el26, &el27, &el28, &el29, &el30, &el31, &el32, &el33, &el34, &el35, &el36,
        &el37, &el38, &el39, &el40, &el41, &el42, &el43, &el44, &el45, &el46, &el47, &el48, &el49, &el50, &el51,
        &el52, &el53, &el54, &el55, &el56, &el57, &el58, &el59, &el60, &el61, &el62, &el63, &el64
    );
    compareExchange64<sortOrder>(
        &el1, &el2, &el3, &el4, &el5, &el6, &el7, &el8, &el9, &el10, &el11, &el12, &el13, &el14, &el15, &el16,
        &el17, &el18, &el19, &el20, &el21, &el22, &el23, &el24, &el25, &el26, &el27, &el28, &el29, &el30, &el31,
        &el32, &el33, &el34, &el35, &el36, &el37, &el38, &el39, &el40, &el41, &el42, &el43, &el44, &el45, &el46,
        &el47, &el48, &el49, &el50, &el51, &el52, &el53, &el54, &el55, &el56, &el57, &el58, &el59, &el60, &el61,
        &el62, &el63, &el64
    );
    store64(
        table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8,
        el9, el10, el11, el12, el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25,
        el26, el27, el28, el29, el30, el31, el32, el33, el34, el35, el36, el37, el38, el39, el40, el41, el42,
        el43, el44, el45, el46, el47, el48, el49, el50, el51, el52, el53, el54, el55, el56, el57, el58, el59,
        el60, el61, el62, el63, el64
    );
}

template __global__ void multiStep6Kernel<ORDER_ASC>(data_t *table, uint_t tableLen, uint_t step);
template __global__ void multiStep6Kernel<ORDER_DESC>(data_t *table, uint_t tableLen, uint_t step);


/*
Global bitonic merge - needed for first step of every phase, when stride is greater than shared memory size.
*/
template <order_t sortOrder>
__global__ void bitonicMergeGlobalKernel(data_t *dataTable, uint_t tableLen, uint_t phase)
{
    uint_t stride = 1 << (phase - 1);
    uint_t pairsPerThreadBlock = (THREADS_PER_GLOBAL_MERGE_KO * ELEMS_PER_THREAD_GLOBAL_MERGE_KO) >> 1;
    uint_t indexGlobal = blockIdx.x * pairsPerThreadBlock + threadIdx.x;

    for (uint_t i = 0; i < ELEMS_PER_THREAD_GLOBAL_MERGE_KO >> 1; i++)
    {
        uint_t indexThread = indexGlobal + i * THREADS_PER_GLOBAL_MERGE_KO;
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
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *dataTable, uint_t tableLen, uint_t step)
{
    extern __shared__ data_t mergeTile[];
    bool firstStepOfPhaseCopy = isFirstStepOfPhase;  // isFirstStepOfPhase is not editable (constant)

    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE_KO * ELEMS_PER_THREAD_LOCAL_MERGE_KO;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
    uint_t pairsPerBlockLength = dataBlockLength >> 1;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE_KO)
    {
        mergeTile[tx] = dataTable[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (uint_t stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        for (uint_t tx = threadIdx.x; tx < pairsPerBlockLength; tx += THREADS_PER_LOCAL_MERGE_KO)
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
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE_KO)
    {
        dataTable[offset + tx] = mergeTile[tx];
    }
}

template __global__ void bitonicMergeLocalKernel<ORDER_ASC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_ASC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, true>(data_t *dataTable, uint_t tableLen, uint_t step);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, false>(data_t *dataTable, uint_t tableLen, uint_t step);
