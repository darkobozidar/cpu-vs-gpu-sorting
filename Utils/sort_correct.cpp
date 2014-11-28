#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "data_types_common.h"
#include "host.h"


/*
Compare function for ASCENDING order needed for C++ sort.
*/
int compareAsc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem1) - *((data_t*)elem2);
}

/*
Compare function for DESCENDING order needed for C++ sort.
*/
int compareDesc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem2) - *((data_t*)elem1);
}

/*
Sorts data with C++ sort, which sorts data 100% correctly. This is needed to verify parallel and sequential sorts.
*/
double sortCorrect(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;

    startStopwatch(&timer);

    if (sortOrder == ORDER_ASC)
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
    }
    else
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
    }

    return endStopwatch(timer);
}
