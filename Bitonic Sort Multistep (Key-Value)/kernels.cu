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
Compares 2 elements and exchanges them according to sortOrder.
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

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
template <order_t sortOrder>
__device__ void compareExchange(data_t *key1, data_t *key2, data_t *val1, data_t *val2)
{
    if ((*key1 > *key2) ^ sortOrder)
    {
        data_t temp = *key1;
        *key1 = *key2;
        *key2 = temp;

        temp = *val1;
        *val1 = *val2;
        *val2 = temp;
    }
}

/*
Compares and exchanges elements according to bitonic sort for 4 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange4(data_t *el1, data_t *el2, data_t *el3, data_t *el4)
{
    // Step n + 1
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);

    // Step n
    compareExchange2<sortOrder>(el1, el3);
    compareExchange2<sortOrder>(el2, el4);
}

/*
Compares and exchanges elements according to bitonic sort for 8 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange8(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8
)
{
    // Step n + 2
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);
    compareExchange2<sortOrder>(el5, el6);
    compareExchange2<sortOrder>(el7, el8);

    // Steps n + 1, n
    compareExchange4<sortOrder>(el1, el5, el3, el7);
    compareExchange4<sortOrder>(el2, el6, el4, el8);
}

/*
Compares and exchanges elements according to bitonic sort for 16 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange16(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8,
    data_t *el9, data_t *el10, data_t *el11, data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16
)
{
    // Step n + 3
    compareExchange2<sortOrder>(el1, el2);
    compareExchange2<sortOrder>(el3, el4);
    compareExchange2<sortOrder>(el5, el6);
    compareExchange2<sortOrder>(el7, el8);
    compareExchange2<sortOrder>(el9, el10);
    compareExchange2<sortOrder>(el11, el12);
    compareExchange2<sortOrder>(el13, el14);
    compareExchange2<sortOrder>(el15, el16);

    // Steps n + 2, n + 1, n
    compareExchange8<sortOrder>(el1, el9, el3, el11, el5, el13, el7, el15);
    compareExchange8<sortOrder>(el2, el10, el4, el12, el6, el14, el8, el16);
}

/*
Compares and exchanges elements according to bitonic sort for 32 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange32(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8,
    data_t *el9, data_t *el10, data_t *el11, data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16,
    data_t *el17, data_t *el18, data_t *el19, data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24,
    data_t *el25, data_t *el26, data_t *el27, data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32
)
{
    // Step n + 4
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

    // Steps n + 3, n + 2, n + 1, n
    compareExchange16<sortOrder>(
        el1, el17, el3, el19, el5, el21, el7, el23, el9, el25, el11, el27, el13, el29, el15, el31
    );
    compareExchange16<sortOrder>(
        el2, el18, el4, el20, el6, el22, el8, el24, el10, el26, el12, el28, el14, el30, el16, el32
    );
}

/*
Compares and exchanges elements according to bitonic sort for 32 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange64(
    data_t *el1, data_t *el2, data_t *el3, data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8,
    data_t *el9, data_t *el10, data_t *el11, data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16,
    data_t *el17, data_t *el18, data_t *el19, data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24,
    data_t *el25, data_t *el26, data_t *el27, data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32,
    data_t *el33, data_t *el34, data_t *el35, data_t *el36, data_t *el37, data_t *el38, data_t *el39, data_t *el40,
    data_t *el41, data_t *el42, data_t *el43, data_t *el44, data_t *el45, data_t *el46, data_t *el47, data_t *el48,
    data_t *el49, data_t *el50, data_t *el51, data_t *el52, data_t *el53, data_t *el54, data_t *el55, data_t *el56,
    data_t *el57, data_t *el58, data_t *el59, data_t *el60, data_t *el61, data_t *el62, data_t *el63, data_t *el64
)
{
    // Step n + 5
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
    compareExchange2<sortOrder>(el33, el34);
    compareExchange2<sortOrder>(el35, el36);
    compareExchange2<sortOrder>(el37, el38);
    compareExchange2<sortOrder>(el39, el40);
    compareExchange2<sortOrder>(el41, el42);
    compareExchange2<sortOrder>(el43, el44);
    compareExchange2<sortOrder>(el45, el46);
    compareExchange2<sortOrder>(el47, el48);
    compareExchange2<sortOrder>(el49, el50);
    compareExchange2<sortOrder>(el51, el52);
    compareExchange2<sortOrder>(el53, el54);
    compareExchange2<sortOrder>(el55, el56);
    compareExchange2<sortOrder>(el57, el58);
    compareExchange2<sortOrder>(el59, el60);
    compareExchange2<sortOrder>(el61, el62);
    compareExchange2<sortOrder>(el63, el64);

    // Steps n + 4, n + 3, n + 2, n + 1, n
    compareExchange32<sortOrder>(
        el1, el33, el3, el35, el5, el37, el7, el39, el9, el41, el11, el43, el13, el45, el15, el47, el17, el49,
        el19, el51, el21, el53, el23, el55, el25, el57, el27, el59, el29, el61, el31, el63
    );
    compareExchange32<sortOrder>(
        el2, el34, el4, el36, el6, el38, el8, el40, el10, el42, el12, el44, el14, el46, el16, el48, el18, el50,
        el20, el52, el22, el54, el24, el56, el26, el58, el28, el60, el30, el62, el32, el64
    );
}


/*
Loads 2 elements if they are inside table length boundaries. In opposite case MIN/MAX value is used
(in order not to influence the sort which follows the load).
*/
template <order_t sortOrder>
__device__ void load2(data_t *table, data_t *tableEnd, uint_t stride, data_t *el1, data_t *el2)
{
    data_t val = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    *el1 = table < tableEnd ? table[0] : val;
    *el2 = table + stride < tableEnd ? table[stride] : val;
}

/*
Stores 2 elements if they are inside table length boundaries.
*/
template <order_t sortOrder>
__device__ void store2(data_t *table, data_t *tableEnd, uint_t stride, data_t el1, data_t el2)
{
    if (table < tableEnd)
    {
        table[0] = el1;
    }
    if (table + stride < tableEnd)
    {
        table[stride] = el2;
    }
}

/*
Loads 4 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load4(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4
)
{
    load2<sortOrder>(table, tableEnd, stride, el1, el2);
    load2<sortOrder>(table + tableOffset, tableEnd, stride, el3, el4);
}

/*
Stores 4 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store4(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4
)
{
    store2<sortOrder>(table, tableEnd, stride, el1, el2);
    store2<sortOrder>(table + tableOffset, tableEnd, stride, el3, el4);
}

/*
Loads 8 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load8(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8
)
{
    load4<sortOrder>(table, tableEnd, tableOffset, stride, el1, el2, el3, el4);
    load4<sortOrder>(table + 2 * tableOffset, tableEnd, tableOffset, stride, el5, el6, el7, el8);
}

/*
Stores 8 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store8(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8
)
{
    store4<sortOrder>(table, tableEnd, tableOffset, stride, el1, el2, el3, el4);
    store4<sortOrder>(table + 2 * tableOffset, tableEnd, tableOffset, stride, el5, el6, el7, el8);
}

/*
Loads 16 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load16(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11,
    data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16
)
{
    load8<sortOrder>(table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    load8<sortOrder>(
        table + 4 * tableOffset, tableEnd, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16
    );
}

/*
Stores 16 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store16(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16
)
{
    store8<sortOrder>(table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    store8<sortOrder>(
        table + 4 * tableOffset, tableEnd, tableOffset, stride, el9, el10, el11, el12, el13, el14, el15, el16
    );
}

/*
Loads 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load32(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11,
    data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16, data_t *el17, data_t *el18, data_t *el19,
    data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24, data_t *el25, data_t *el26, data_t *el27,
    data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32
)
{
    load16<sortOrder>(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16
    );
    load16<sortOrder>(
        table + 8 * tableOffset, tableEnd, tableOffset, stride, el17, el18, el19, el20, el21, el22, el23,
        el24, el25, el26, el27, el28, el29, el30, el31, el32
    );
}

/*
Stores 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store32(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16, data_t el17, data_t el18, data_t el19,
    data_t el20, data_t el21, data_t el22, data_t el23, data_t el24, data_t el25, data_t el26, data_t el27,
    data_t el28, data_t el29, data_t el30, data_t el31, data_t el32
)
{
    store16<sortOrder>(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16
    );
    store16<sortOrder>(
        table + 8 * tableOffset, tableEnd, tableOffset, stride, el17, el18, el19, el20, el21, el22, el23,
        el24, el25, el26, el27, el28, el29, el30, el31, el32
    );
}

/*
Loads 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load64(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t *el1, data_t *el2, data_t *el3,
    data_t *el4, data_t *el5, data_t *el6, data_t *el7, data_t *el8, data_t *el9, data_t *el10, data_t *el11,
    data_t *el12, data_t *el13, data_t *el14, data_t *el15, data_t *el16, data_t *el17, data_t *el18, data_t *el19,
    data_t *el20, data_t *el21, data_t *el22, data_t *el23, data_t *el24, data_t *el25, data_t *el26, data_t *el27,
    data_t *el28, data_t *el29, data_t *el30, data_t *el31, data_t *el32, data_t *el33, data_t *el34, data_t *el35,
    data_t *el36, data_t *el37, data_t *el38, data_t *el39, data_t *el40, data_t *el41, data_t *el42, data_t *el43,
    data_t *el44, data_t *el45, data_t *el46, data_t *el47, data_t *el48, data_t *el49, data_t *el50, data_t *el51,
    data_t *el52, data_t *el53, data_t *el54, data_t *el55, data_t *el56, data_t *el57, data_t *el58, data_t *el59,
    data_t *el60, data_t *el61, data_t *el62, data_t *el63, data_t *el64
)
{
    load32<sortOrder>(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29,
        el30, el31, el32
    );
    load32<sortOrder>(
        table + 16 * tableOffset, tableEnd, tableOffset, stride, el33, el34, el35, el36, el37, el38, el39, el40,
        el41, el42, el43, el44, el45, el46, el47, el48, el49, el50, el51, el52, el53, el54, el55, el56, el57,
        el58, el59, el60, el61, el62, el63, el64
    );
}

/*
Stores 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store64(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16, data_t el17, data_t el18, data_t el19,
    data_t el20, data_t el21, data_t el22, data_t el23, data_t el24, data_t el25, data_t el26, data_t el27,
    data_t el28, data_t el29, data_t el30, data_t el31, data_t el32, data_t el33, data_t el34, data_t el35,
    data_t el36, data_t el37, data_t el38, data_t el39, data_t el40, data_t el41, data_t el42, data_t el43,
    data_t el44, data_t el45, data_t el46, data_t el47, data_t el48, data_t el49, data_t el50, data_t el51,
    data_t el52, data_t el53, data_t el54, data_t el55, data_t el56, data_t el57, data_t el58, data_t el59,
    data_t el60, data_t el61, data_t el62, data_t el63, data_t el64
)
{
    store32<sortOrder>(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29,
        el30, el31, el32
    );
    store32<sortOrder>(
        table + 16 * tableOffset, tableEnd, tableOffset, stride, el33, el34, el35, el36, el37, el38, el39, el40,
        el41, el42, el43, el44, el45, el46, el47, el48, el49, el50, el51, el52, el53, el54, el55, el56, el57,
        el58, el59, el60, el61, el62, el63, el64
    );
}


/*
Generates parameters needed for multistep bitonic sort.
> stride - (gap) between two elements beeing compared
> threadsPerSubBlocks - how many threads apper per sub-block in current step
> indexTable - start index, at which thread should start reading elements
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
__global__ void bitonicSortKernel(data_t *keys, data_t *values, uint_t tableLen)
{
    extern __shared__ data_t bitonicSortTile[];

    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;

    data_t *keysTile = bitonicSortTile;
    data_t *valuesTile = bitonicSortTile + dataBlockLength;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT)
    {
        keysTile[tx] = keys[offset + tx];
        valuesTile[tx] = values[offset + tx];
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

                compareExchange<sortOrder>(
                    &keysTile[index], &keysTile[index + offset], &valuesTile[index], &valuesTile[index + offset]
                );
            }

            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_BITONIC_SORT) {
        keys[offset + tx] = keysTile[tx];
        values[offset + tx] = valuesTile[tx];
    }
}

template __global__ void bitonicSortKernel<ORDER_ASC>(data_t *keys, data_t *values, uint_t tableLen);
template __global__ void bitonicSortKernel<ORDER_DESC>(data_t *keys, data_t *values, uint_t tableLen);


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
    store2<sortOrder>(table + indexTable, table + tableLen, stride, el1, el2);
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
    store4<sortOrder>(table + indexTable, table + tableLen, tableOffset, stride, el1, el2, el3, el4);
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
    store8<sortOrder>(
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
    store16<sortOrder>(
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
    store32<sortOrder>(
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
    store64<sortOrder>(
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
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
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
