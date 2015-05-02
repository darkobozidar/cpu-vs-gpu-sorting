#ifndef KERNELS_KEY_ONLY_BITONIC_SORT_MULTISTEP_H
#define KERNELS_KEY_ONLY_BITONIC_SORT_MULTISTEP_H


#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "common_utils.h"
#include "key_only_utils.h"


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
    compareExchange<sortOrder>(&el1, &el2);
    store2(table + indexTable, table + tableLen, stride, el1, el2);
}

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

#endif
