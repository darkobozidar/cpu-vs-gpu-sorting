#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "data_types_common.h"
#include "host.h"

/*
Compare function for ASCENDING order needed for C++ qsort.
*/
int compareAsc(const void* elem1, const void* elem2)
{
    // Cannot use subtraction because of unsigned data types. Another option would be to convert to bigger data
    // type, but the result has to be converted to int.
    if (*((data_t*)elem1) > *((data_t*)elem2))
    {
        return 1;
    }
    else if (*((data_t*)elem1) < *((data_t*)elem2))
    {
        return -1;
    }

    return 0;
}

/*
Compare function for DESCENDING order needed for C++ qsort.
*/
int compareDesc(const void* elem1, const void* elem2)
{
    // Cannot use subtraction because of unsigned data types. Another option would be to convert to bigger data
    // type, but the result has to be converted to int.
    if (*((data_t*)elem1) < *((data_t*)elem2))
    {
        return 1;
    }
    else if (*((data_t*)elem1) > *((data_t*)elem2))
    {
        return -1;
    }

    return 0;
}

/*
Sorts an array with C quicksort implementation.
*/
template <typename T>
void quickSort(T *dataTable, uint_t tableLen, order_t sortOrder)
{
    if (sortOrder == ORDER_ASC)
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
    }
    else
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
    }
}

template void quickSort<data_t>(data_t *dataTable, uint_t tableLen, order_t sortOrder);
template void quickSort<uint_t>(uint_t *dataTable, uint_t tableLen, order_t sortOrder);
template void quickSort<int_t>(int_t *dataTable, uint_t tableLen, order_t sortOrder);


/*
Sorts data with C++ vector sort.
*/
template <typename T>
void stdVectorSort(T *dataTable, uint_t tableLen, order_t sortOrder)
{
    std::vector<T> dataVector(dataTable, dataTable + tableLen);

    if (sortOrder == ORDER_ASC)
    {
        std::sort(dataVector.begin(), dataVector.end());
    }
    else
    {
        std::sort(dataVector.rbegin(), dataVector.rend());
    }

    std::copy(dataVector.begin(), dataVector.end(), dataTable);
}

template void stdVectorSort<data_t>(data_t *dataTable, uint_t tableLen, order_t sortOrder);
template void stdVectorSort<uint_t>(uint_t *dataTable, uint_t tableLen, order_t sortOrder);
template void stdVectorSort<int_t>(int_t *dataTable, uint_t tableLen, order_t sortOrder);


/*
Sorts data with C++ qsort, which sorts data 100% correctly. This is needed to verify parallel and sequential
sorts.
*/
double sortCorrect(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    // C++ std vector sort is faster than C++ Quicksort. But vector sort throws exception, if too much memory
    // is allocated. For example a lot of arrays are created in "sample sort key-value". In that case C++ vector
    // sort throws exception, if array length is more or equal than "2^24".
    stdVectorSort<data_t>(dataTable, tableLen, sortOrder);
    //quickSort<data_t>(dataTable, tableLen, sortOrder);

    return endStopwatch(timer);
}
