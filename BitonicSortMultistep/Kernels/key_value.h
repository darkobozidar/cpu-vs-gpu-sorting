#ifndef KERNELS_KEY_VALUE_BITONIC_SORT_MULTISTEP_H
#define KERNELS_KEY_VALUE_BITONIC_SORT_MULTISTEP_H


#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "common_utils.h"
#include "key_value_utils.h"


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
    compareExchange<sortOrder>(&key1, &key2, &val1, &val2);
    store2(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, stride, key1, key2, val1, val2
    );
}

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
    store4(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride,
        key1, key2, key3, key4, val1, val2, val3, val4
    );
}

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
    store8(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, val1, val2, val3, val4, val5, val6, val7, val8
    );
}

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
    store16(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16, val1, val2,
        val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16
    );
}

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
    store32(
        keys + indexTable, values + indexTable, tableLen - indexTable - 1, tableOffset, stride, key1, key2,
        key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16, key17,
        key18, key19, key20, key21, key22, key23, key24, key25, key26, key27, key28, key29, key30, key31,
        key32, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14,
        val15, val16, val17, val18, val19, val20, val21, val22, val23, val24, val25, val26, val27, val28,
        val29, val30, val31, val32
    );
}

#endif
