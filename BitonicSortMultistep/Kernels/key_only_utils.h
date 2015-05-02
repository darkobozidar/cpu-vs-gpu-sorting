#ifndef KERNELS_KEY_ONLY_UTILS_BITONIC_SORT_MULTISTEP_H
#define KERNELS_KEY_ONLY_UTILS_BITONIC_SORT_MULTISTEP_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"


/*
Compares and exchanges elements according to bitonic sort for 4 elements.
*/
template <order_t sortOrder>
__device__ void compareExchange4(data_t *el1, data_t *el2, data_t *el3, data_t *el4)
{
    // Step n + 1
    compareExchange<sortOrder>(el1, el2);
    compareExchange<sortOrder>(el3, el4);

    // Step n
    compareExchange<sortOrder>(el1, el3);
    compareExchange<sortOrder>(el2, el4);
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
    compareExchange<sortOrder>(el1, el2);
    compareExchange<sortOrder>(el3, el4);
    compareExchange<sortOrder>(el5, el6);
    compareExchange<sortOrder>(el7, el8);

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
    compareExchange<sortOrder>(el1, el2);
    compareExchange<sortOrder>(el3, el4);
    compareExchange<sortOrder>(el5, el6);
    compareExchange<sortOrder>(el7, el8);
    compareExchange<sortOrder>(el9, el10);
    compareExchange<sortOrder>(el11, el12);
    compareExchange<sortOrder>(el13, el14);
    compareExchange<sortOrder>(el15, el16);

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
    compareExchange<sortOrder>(el1, el2);
    compareExchange<sortOrder>(el3, el4);
    compareExchange<sortOrder>(el5, el6);
    compareExchange<sortOrder>(el7, el8);
    compareExchange<sortOrder>(el9, el10);
    compareExchange<sortOrder>(el11, el12);
    compareExchange<sortOrder>(el13, el14);
    compareExchange<sortOrder>(el15, el16);
    compareExchange<sortOrder>(el17, el18);
    compareExchange<sortOrder>(el19, el20);
    compareExchange<sortOrder>(el21, el22);
    compareExchange<sortOrder>(el23, el24);
    compareExchange<sortOrder>(el25, el26);
    compareExchange<sortOrder>(el27, el28);
    compareExchange<sortOrder>(el29, el30);
    compareExchange<sortOrder>(el31, el32);

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
    compareExchange<sortOrder>(el1, el2);
    compareExchange<sortOrder>(el3, el4);
    compareExchange<sortOrder>(el5, el6);
    compareExchange<sortOrder>(el7, el8);
    compareExchange<sortOrder>(el9, el10);
    compareExchange<sortOrder>(el11, el12);
    compareExchange<sortOrder>(el13, el14);
    compareExchange<sortOrder>(el15, el16);
    compareExchange<sortOrder>(el17, el18);
    compareExchange<sortOrder>(el19, el20);
    compareExchange<sortOrder>(el21, el22);
    compareExchange<sortOrder>(el23, el24);
    compareExchange<sortOrder>(el25, el26);
    compareExchange<sortOrder>(el27, el28);
    compareExchange<sortOrder>(el29, el30);
    compareExchange<sortOrder>(el31, el32);
    compareExchange<sortOrder>(el33, el34);
    compareExchange<sortOrder>(el35, el36);
    compareExchange<sortOrder>(el37, el38);
    compareExchange<sortOrder>(el39, el40);
    compareExchange<sortOrder>(el41, el42);
    compareExchange<sortOrder>(el43, el44);
    compareExchange<sortOrder>(el45, el46);
    compareExchange<sortOrder>(el47, el48);
    compareExchange<sortOrder>(el49, el50);
    compareExchange<sortOrder>(el51, el52);
    compareExchange<sortOrder>(el53, el54);
    compareExchange<sortOrder>(el55, el56);
    compareExchange<sortOrder>(el57, el58);
    compareExchange<sortOrder>(el59, el60);
    compareExchange<sortOrder>(el61, el62);
    compareExchange<sortOrder>(el63, el64);

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
    if (table < tableEnd)
    {
        *el1 = table[0];
    }
    else
    {
        *el1 = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    }

    if (table + stride < tableEnd)
    {
        *el2 = table[stride];
    }
    else
    {
        *el2 = sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
    }
}

/*
Stores 2 elements if they are inside table length boundaries.
*/
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
__device__ void store4(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4
)
{
    store2(table, tableEnd, stride, el1, el2);
    store2(table + tableOffset, tableEnd, stride, el3, el4);
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
__device__ void store8(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8
)
{
    store4(table, tableEnd, tableOffset, stride, el1, el2, el3, el4);
    store4(table + 2 * tableOffset, tableEnd, tableOffset, stride, el5, el6, el7, el8);
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
__device__ void store16(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16
)
{
    store8(table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8);
    store8(
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
__device__ void store32(
    data_t *table, data_t *tableEnd, uint_t tableOffset, uint_t stride, data_t el1, data_t el2, data_t el3,
    data_t el4, data_t el5, data_t el6, data_t el7, data_t el8, data_t el9, data_t el10, data_t el11,
    data_t el12, data_t el13, data_t el14, data_t el15, data_t el16, data_t el17, data_t el18, data_t el19,
    data_t el20, data_t el21, data_t el22, data_t el23, data_t el24, data_t el25, data_t el26, data_t el27,
    data_t el28, data_t el29, data_t el30, data_t el31, data_t el32
)
{
    store16(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16
    );
    store16(
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
    store32(
        table, tableEnd, tableOffset, stride, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12,
        el13, el14, el15, el16, el17, el18, el19, el20, el21, el22, el23, el24, el25, el26, el27, el28, el29,
        el30, el31, el32
    );
    store32(
        table + 16 * tableOffset, tableEnd, tableOffset, stride, el33, el34, el35, el36, el37, el38, el39, el40,
        el41, el42, el43, el44, el45, el46, el47, el48, el49, el50, el51, el52, el53, el54, el55, el56, el57,
        el58, el59, el60, el61, el62, el63, el64
    );
}

#endif
