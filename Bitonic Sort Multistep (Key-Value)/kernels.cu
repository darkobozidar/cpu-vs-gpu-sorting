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
__device__ void compareExchange2(data_t *key1, data_t *key2, data_t *val1, data_t *val2)
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
__device__ void compareExchange4(
    data_t *key1, data_t *key2, data_t *key3, data_t *key4, data_t *val1, data_t *val2, data_t *val3, data_t *val4
)
{
    // Step n + 1
    compareExchange2<sortOrder>(key1, key2, val1, val2);
    compareExchange2<sortOrder>(key3, key4, val3, val4);

    // Step n
    compareExchange2<sortOrder>(key1, key3, val1, val3);
    compareExchange2<sortOrder>(key2, key4, val2, val4);
}

/*
Compares and exchanges elements according to bitonic sort for 8 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange8(
    data_t *key1, data_t *key2, data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7, data_t *key8,
    data_t *val1, data_t *val2, data_t *val3, data_t *val4, data_t *val5, data_t *val6, data_t *val7, data_t *val8
)
{
    // Step n + 2
    compareExchange2<sortOrder>(key1, key2, val1, val2);
    compareExchange2<sortOrder>(key3, key4, val3, val4);
    compareExchange2<sortOrder>(key5, key6, val5, val6);
    compareExchange2<sortOrder>(key7, key8, val7, val8);

    // Steps n + 1, n
    compareExchange4<sortOrder>(key1, key5, key3, key7, val1, val5, val3, val7);
    compareExchange4<sortOrder>(key2, key6, key4, key8, val2, val6, val4, val8);
}

/*
Compares and exchanges elements according to bitonic sort for 16 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange16(
    data_t *key1, data_t *key2, data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7,
    data_t *key8, data_t *key9, data_t *key10, data_t *key11, data_t *key12, data_t *key13, data_t *key14,
    data_t *key15, data_t *key16, data_t *val1, data_t *val2, data_t *val3, data_t *val4, data_t *val5,
    data_t *val6, data_t *val7, data_t *val8, data_t *val9, data_t *val10, data_t *val11, data_t *val12,
    data_t *val13, data_t *val14, data_t *val15, data_t *val16
)
{
    // Step n + 3
    compareExchange2<sortOrder>(key1, key2, val1, val2);
    compareExchange2<sortOrder>(key3, key4, val3, val4);
    compareExchange2<sortOrder>(key5, key6, val5, val6);
    compareExchange2<sortOrder>(key7, key8, val7, val8);
    compareExchange2<sortOrder>(key9, key10, val9, val10);
    compareExchange2<sortOrder>(key11, key12, val11, val12);
    compareExchange2<sortOrder>(key13, key14, val13, val14);
    compareExchange2<sortOrder>(key15, key16, val15, val16);

    // Steps n + 2, n + 1, n
    compareExchange8<sortOrder>(
        key1, key9, key3, key11, key5, key13, key7, key15, val1, val9, val3, val11, val5, val13, val7, val15
    );
    compareExchange8<sortOrder>(
        key2, key10, key4, key12, key6, key14, key8, key16, val2, val10, val4, val12, val6, val14, val8, val16
    );
}

/*
Compares and exchanges elements according to bitonic sort for 32 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange32(
    data_t *key1, data_t *key2, data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7,
    data_t *key8, data_t *key9, data_t *key10, data_t *key11, data_t *key12, data_t *key13, data_t *key14,
    data_t *key15, data_t *key16, data_t *key17, data_t *key18, data_t *key19, data_t *key20, data_t *key21,
    data_t *key22, data_t *key23, data_t *key24, data_t *key25, data_t *key26, data_t *key27, data_t *key28,
    data_t *key29, data_t *key30, data_t *key31, data_t *key32, data_t *val1, data_t *val2, data_t *val3,
    data_t *val4, data_t *val5, data_t *val6, data_t *val7, data_t *val8, data_t *val9, data_t *val10,
    data_t *val11, data_t *val12, data_t *val13, data_t *val14, data_t *val15, data_t *val16, data_t *val17,
    data_t *val18, data_t *val19, data_t *val20, data_t *val21, data_t *val22, data_t *val23, data_t *val24,
    data_t *val25, data_t *val26, data_t *val27, data_t *val28, data_t *val29, data_t *val30, data_t *val31,
    data_t *val32
)
{
    // Step n + 4
    compareExchange2<sortOrder>(key1, key2, val1, val2);
    compareExchange2<sortOrder>(key3, key4, val3, val4);
    compareExchange2<sortOrder>(key5, key6, val5, val6);
    compareExchange2<sortOrder>(key7, key8, val7, val8);
    compareExchange2<sortOrder>(key9, key10, val9, val10);
    compareExchange2<sortOrder>(key11, key12, val11, val12);
    compareExchange2<sortOrder>(key13, key14, val13, val14);
    compareExchange2<sortOrder>(key15, key16, val15, val16);
    compareExchange2<sortOrder>(key17, key18, val17, val18);
    compareExchange2<sortOrder>(key19, key20, val19, val20);
    compareExchange2<sortOrder>(key21, key22, val21, val22);
    compareExchange2<sortOrder>(key23, key24, val23, val24);
    compareExchange2<sortOrder>(key25, key26, val25, val26);
    compareExchange2<sortOrder>(key27, key28, val27, val28);
    compareExchange2<sortOrder>(key29, key30, val29, val30);
    compareExchange2<sortOrder>(key31, key32, val31, val32);

    // Steps n + 3, n + 2, n + 1, n
    compareExchange16<sortOrder>(
        key1, key17, key3, key19, key5, key21, key7, key23, key9, key25, key11, key27, key13, key29, key15, key31,
        val1, val17, val3, val19, val5, val21, val7, val23, val9, val25, val11, val27, val13, val29, val15, val31
    );
    compareExchange16<sortOrder>(
        key2, key18, key4, key20, key6, key22, key8, key24, key10, key26, key12, key28, key14, key30, key16, key32,
        val2, val18, val4, val20, val6, val22, val8, val24, val10, val26, val12, val28, val14, val30, val16, val32
    );
}


/*
Loads 2 elements if they are inside table length boundaries. In opposite case MIN/MAX value is used
(in order not to influence the sort which follows the load).
*/
template <order_t sortOrder>
__device__ void load2(
    data_t *keys, data_t *values, int_t tableLen, int_t stride, data_t *key1, data_t *key2, data_t *val1, data_t *val2
)
{
    if (tableLen >= 0)
    {
        *key1 = keys[0];
        *val1 = values[0];
    }
    else
    {
        // Value is not important
        *key1 = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    }

    if (tableLen >= stride)
    {
        *key2 = keys[stride];
        *val2 = values[stride];
    }
    else
    {
        // Value is not important
        *key2 = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    }
}

/*
Stores 2 elements if they are inside table length boundaries.
*/
template <order_t sortOrder>
__device__ void store2(
    data_t *keys, data_t *values, int_t tableLen, int_t stride, data_t key1, data_t key2, data_t val1, data_t val2
)
{
    if (tableLen >= 0)
    {
        keys[0] = key1;
        values[0] = val1;
    }

    if (tableLen >= stride)
    {
        keys[stride] = key2;
        values[stride] = val2;
    }
}

/*
Loads 4 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load4(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t *key1, data_t *key2,
    data_t *key3, data_t *key4, data_t *val1, data_t *val2, data_t *val3, data_t *val4
)
{
    load2<sortOrder>(keys, values, tableLen, stride, key1, key2, val1, val2);
    load2<sortOrder>(
        keys + tableOffset, values + tableOffset, tableLen - tableOffset, stride, key3, key4, val3, val4
    );
}

/*
Stores 4 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store4(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t key1, data_t key2,
    data_t key3, data_t key4, data_t val1, data_t val2, data_t val3, data_t val4
)
{
    store2<sortOrder>(keys, values, tableLen, stride, key1, key2, val1, val2);
    store2<sortOrder>(
        keys + tableOffset, values + tableOffset, tableLen - tableOffset, stride, key3, key4, val3, val4
    );
}

/*
Loads 8 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load8(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t *key1, data_t *key2,
    data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7, data_t *key8, data_t *val1, data_t *val2,
    data_t *val3, data_t *val4, data_t *val5, data_t *val6, data_t *val7, data_t *val8
)
{
    load4<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, val1, val2, val3, val4
    );
    load4<sortOrder>(
        keys + 2 * tableOffset, values + 2 * tableOffset, tableLen - 2 * tableOffset, tableOffset, stride,
        key5, key6, key7, key8, val5, val6, val7, val8
    );
}

/*
Stores 8 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store8(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t key1, data_t key2,
    data_t key3, data_t key4, data_t key5, data_t key6, data_t key7, data_t key8, data_t val1, data_t val2,
    data_t val3, data_t val4, data_t val5, data_t val6, data_t val7, data_t val8
)
{
    store4<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, val1, val2, val3, val4
    );
    store4<sortOrder>(
        keys + 2 * tableOffset, values + 2 * tableOffset, tableLen - 2 * tableOffset, tableOffset, stride,
        key5, key6, key7, key8, val5, val6, val7, val8
    );
}

/*
Loads 16 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load16(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t *key1, data_t *key2,
    data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7, data_t *key8, data_t *key9, data_t *key10,
    data_t *key11, data_t *key12, data_t *key13, data_t *key14, data_t *key15, data_t *key16, data_t *val1,
    data_t *val2, data_t *val3, data_t *val4, data_t *val5, data_t *val6, data_t *val7, data_t *val8, data_t *val9,
    data_t *val10, data_t *val11, data_t *val12, data_t *val13, data_t *val14, data_t *val15, data_t *val16
)
{
    load8<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, key5, key6, key7, key8,
        val1, val2, val3, val4, val5, val6, val7, val8
    );
    load8<sortOrder>(
        keys + 4 * tableOffset, values + 4 * tableOffset, tableLen - 4 * tableOffset, tableOffset, stride,
        key9, key10, key11, key12, key13, key14, key15, key16, val9, val10, val11, val12, val13, val14, val15, val16
    );
}

/*
Stores 16 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store16(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t key1, data_t key2,
    data_t key3, data_t key4, data_t key5, data_t key6, data_t key7, data_t key8, data_t key9, data_t key10,
    data_t key11, data_t key12, data_t key13, data_t key14, data_t key15, data_t key16, data_t val1,
    data_t val2, data_t val3, data_t val4, data_t val5, data_t val6, data_t val7, data_t val8, data_t val9,
    data_t val10, data_t val11, data_t val12, data_t val13, data_t val14, data_t val15, data_t val16
)
{
    store8<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, key5, key6, key7, key8,
        val1, val2, val3, val4, val5, val6, val7, val8
    );
    store8<sortOrder>(
        keys + 4 * tableOffset, values + 4 * tableOffset, tableLen - 4 * tableOffset, tableOffset, stride,
        key9, key10, key11, key12, key13, key14, key15, key16, val9, val10, val11, val12, val13, val14, val15, val16
    );
}

/*
Loads 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void load32(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t *key1, data_t *key2,
    data_t *key3, data_t *key4, data_t *key5, data_t *key6, data_t *key7, data_t *key8, data_t *key9,
    data_t *key10, data_t *key11, data_t *key12, data_t *key13, data_t *key14, data_t *key15, data_t *key16,
    data_t *key17, data_t *key18, data_t *key19, data_t *key20, data_t *key21, data_t *key22, data_t *key23,
    data_t *key24, data_t *key25, data_t *key26, data_t *key27, data_t *key28, data_t *key29, data_t *key30,
    data_t *key31, data_t *key32, data_t *val1, data_t *val2, data_t *val3, data_t *val4, data_t *val5,
    data_t *val6, data_t *val7, data_t *val8, data_t *val9, data_t *val10, data_t *val11, data_t *val12,
    data_t *val13, data_t *val14, data_t *val15, data_t *val16, data_t *val17, data_t *val18, data_t *val19,
    data_t *val20, data_t *val21, data_t *val22, data_t *val23, data_t *val24, data_t *val25, data_t *val26,
    data_t *val27, data_t *val28, data_t *val29, data_t *val30, data_t *val31, data_t *val32
)
{
    load16<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, key5, key6, key7, key8, key9, key10,
        key11, key12, key13, key14, key15, key16, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
        val11, val12, val13, val14, val15, val16
    );
    load16<sortOrder>(
        keys + 8 * tableOffset, values + 8 * tableOffset, tableLen - 8 * tableOffset, tableOffset, stride, key17,
        key18, key19, key20, key21, key22, key23, key24, key25, key26, key27, key28, key29, key30, key31, key32,
        val17, val18, val19, val20, val21, val22, val23, val24, val25, val26, val27, val28, val29, val30, val31, val32
    );
}

/*
Stores 32 elements according to bitonic sort indexes.
*/
template <order_t sortOrder>
__device__ void store32(
    data_t *keys, data_t *values, int_t tableLen, uint_t tableOffset, int_t stride, data_t key1, data_t key2,
    data_t key3, data_t key4, data_t key5, data_t key6, data_t key7, data_t key8, data_t key9, data_t key10,
    data_t key11, data_t key12, data_t key13, data_t key14, data_t key15, data_t key16, data_t key17,
    data_t key18, data_t key19, data_t key20, data_t key21, data_t key22, data_t key23, data_t key24,
    data_t key25, data_t key26, data_t key27, data_t key28, data_t key29, data_t key30, data_t key31,
    data_t key32, data_t val1, data_t val2, data_t val3, data_t val4, data_t val5, data_t val6, data_t val7,
    data_t val8, data_t val9, data_t val10, data_t val11, data_t val12, data_t val13, data_t val14, data_t val15,
    data_t val16, data_t val17, data_t val18, data_t val19, data_t val20, data_t val21, data_t val22,
    data_t val23, data_t val24, data_t val25, data_t val26, data_t val27, data_t val28, data_t val29,
    data_t val30, data_t val31, data_t val32
)
{
    store16<sortOrder>(
        keys, values, tableLen, tableOffset, stride, key1, key2, key3, key4, key5, key6, key7, key8, key9, key10,
        key11, key12, key13, key14, key15, key16, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
        val11, val12, val13, val14, val15, val16
    );
    store16<sortOrder>(
        keys + 8 * tableOffset, values + 8 * tableOffset, tableLen - 8 * tableOffset, tableOffset, stride, key17,
        key18, key19, key20, key21, key22, key23, key24, key25, key26, key27, key28, key29, key30, key31, key32,
        val17, val18, val19, val20, val21, val22, val23, val24, val25, val26, val27, val28, val29, val30, val31, val32
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

                compareExchange2<sortOrder>(
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
__global__ void multiStep1Kernel(data_t *keys, data_t *values, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t key1, key2;
    data_t val1, val2;

    getMultiStepParams(step, 1, stride, tableOffset, indexTable);

    load2<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, stride, &key1, &key2, &val1, &val2
    );
    compareExchange2<sortOrder>(&key1, &key2, &val1, &val2);
    store2<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, stride, key1, key2, val1, val2
    );
}

template __global__ void multiStep1Kernel<ORDER_ASC>(data_t *key, data_t *values, int_t tableLen, uint_t step);
template __global__ void multiStep1Kernel<ORDER_DESC>(data_t *key, data_t *values, int_t tableLen, uint_t step);


/*
Performs bitonic merge with 2-multistep (sorts 4 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep2Kernel(data_t *keys, data_t *values, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t key1, key2, key3, key4;
    data_t val1, val2, val3, val4;

    getMultiStepParams(step, 2, stride, tableOffset, indexTable);

    load4<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride,
        &key1, &key2, &key3, &key4, &val1, &val2, &val3, &val4
    );
    compareExchange4<sortOrder>(&key1, &key2, &key3, &key4, &val1, &val2, &val3, &val4);
    store4<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride,
        key1, key2, key3, key4, val1, val2, val3, val4
    );
}

template __global__ void multiStep2Kernel<ORDER_ASC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);
template __global__ void multiStep2Kernel<ORDER_DESC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);


/*
Performs bitonic merge with 3-multistep (sorts 8 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep3Kernel(data_t *keys, data_t *values, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t key1, key2, key3, key4, key5, key6, key7, key8;
    data_t val1, val2, val3, val4, val5, val6, val7, val8;

    getMultiStepParams(step, 3, stride, tableOffset, indexTable);

    load8<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, &key1, &key2,
        &key3, &key4, &key5, &key6, &key7, &key8, &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8
    );
    compareExchange8<sortOrder>(
        &key1, &key2, &key3, &key4, &key5, &key6, &key7, &key8,
        &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8
    );
    store8<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, val1, val2, val3, val4, val5, val6, val7, val8
    );
}

template __global__ void multiStep3Kernel<ORDER_ASC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);
template __global__ void multiStep3Kernel<ORDER_DESC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);


/*
Performs bitonic merge with 4-multistep (sorts 16 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep4Kernel(data_t *keys, data_t *values, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16;
    data_t val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;

    getMultiStepParams(step, 4, stride, tableOffset, indexTable);

    load16<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, &key1, &key2,
        &key3, &key4, &key5, &key6, &key7, &key8, &key9, &key10, &key11, &key12, &key13, &key14, &key15, &key16,
        &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8, &val9, &val10, &val11, &val12, &val13, &val14,
        &val15, &val16
    );
    compareExchange16<sortOrder>(
        &key1, &key2, &key3, &key4, &key5, &key6, &key7, &key8, &key9, &key10, &key11, &key12, &key13, &key14,
        &key15, &key16, &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8, &val9, &val10, &val11, &val12,
        &val13, &val14, &val15, &val16
    );
    store16<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16, val1, val2,
        val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16
    );
}

template __global__ void multiStep4Kernel<ORDER_ASC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);
template __global__ void multiStep4Kernel<ORDER_DESC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);


/*
Performs bitonic merge with 5-multistep (sorts 32 elements per thread).
*/
template <order_t sortOrder>
__global__ void multiStep5Kernel(data_t *keys, data_t *values, int_t tableLen, uint_t step)
{
    uint_t stride, tableOffset, indexTable;
    data_t key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16,
        key17, key18, key19, key20, key21, key22, key23, key24, key25, key26, key27, key28, key29, key30, key31, key32;
    data_t val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16,
        val17, val18, val19, val20, val21, val22, val23, val24, val25, val26, val27, val28, val29, val30, val31, val32;

    getMultiStepParams(step, 5, stride, tableOffset, indexTable);

    load32<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, &key1, &key2,
        &key3, &key4, &key5, &key6, &key7, &key8, &key9, &key10, &key11, &key12, &key13, &key14, &key15, &key16,
        &key17, &key18, &key19, &key20, &key21, &key22, &key23, &key24, &key25, &key26, &key27, &key28, &key29,
        &key30, &key31, &key32, &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8, &val9, &val10, &val11,
        &val12, &val13, &val14, &val15, &val16, &val17, &val18, &val19, &val20, &val21, &val22, &val23, &val24,
        &val25, &val26, &val27, &val28, &val29, &val30, &val31, &val32
    );
    compareExchange32<sortOrder>(
        &key1, &key2, &key3, &key4, &key5, &key6, &key7, &key8, &key9, &key10, &key11, &key12, &key13, &key14,
        &key15, &key16, &key17, &key18, &key19, &key20, &key21, &key22, &key23, &key24, &key25, &key26, &key27,
        &key28, &key29, &key30, &key31, &key32, &val1, &val2, &val3, &val4, &val5, &val6, &val7, &val8, &val9,
        &val10, &val11, &val12, &val13, &val14, &val15, &val16, &val17, &val18, &val19, &val20, &val21, &val22,
        &val23, &val24, &val25, &val26, &val27, &val28, &val29, &val30, &val31, &val32
    );
    store32<sortOrder>(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16, key17,
        key18, key19, key20, key21, key22, key23, key24, key25, key26, key27, key28, key29, key30, key31,
        key32, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14,
        val15, val16, val17, val18, val19, val20, val21, val22, val23, val24, val25, val26, val27, val28,
        val29, val30, val31, val32
    );
}

template __global__ void multiStep5Kernel<ORDER_ASC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);
template __global__ void multiStep5Kernel<ORDER_DESC>(data_t *keys, data_t *values, int_t tableLen, uint_t step);


/*
Global bitonic merge - needed for first step of every phase, when stride is greater than shared memory size.
*/
template <order_t sortOrder>
__global__ void bitonicMergeGlobalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
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

        compareExchange2<sortOrder>(&keys[index], &keys[index + offset], &values[index], &values[index + offset]);
    }
}

template __global__ void bitonicMergeGlobalKernel<ORDER_ASC>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);
template __global__ void bitonicMergeGlobalKernel<ORDER_DESC>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);


/*
Local bitonic merge for sections, where stride IS LOWER OR EQUAL than max shared memory.
*/
template <order_t sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t step)
{
    extern __shared__ data_t mergeTile[];
    bool firstStepOfPhaseCopy = isFirstStepOfPhase;  // isFirstStepOfPhase is not editable (constant)

    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;
    uint_t offset = blockIdx.x * elemsPerThreadBlock;
    uint_t dataBlockLength = offset + elemsPerThreadBlock <= tableLen ? elemsPerThreadBlock : tableLen - offset;
    uint_t pairsPerBlockLength = dataBlockLength >> 1;

    data_t *keysTile = mergeTile;
    data_t *valuesTile = mergeTile + dataBlockLength;

    // Reads data from global to shared memory.
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        keysTile[tx] = keys[offset + tx];
        valuesTile[tx] = values[offset + tx];
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

            compareExchange2<sortOrder>(
                &keysTile[index], &keysTile[index + offset], &valuesTile[index], &valuesTile[index + offset]
            );
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (uint_t tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PER_LOCAL_MERGE)
    {
        keys[offset + tx] = keysTile[tx];
        values[offset + tx] = valuesTile[tx];
    }
}

template __global__ void bitonicMergeLocalKernel<ORDER_ASC, true>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);
template __global__ void bitonicMergeLocalKernel<ORDER_ASC, false>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, true>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);
template __global__ void bitonicMergeLocalKernel<ORDER_DESC, false>(
    data_t *keys, data_t *values, uint_t tableLen, uint_t step
);