#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "data_types_common.h"
#include "host.h"


/*
Sorts data with C++ sort, which sorts data 100% correctly. This is needed to verify parallel and sequential sorts.
*/
double sortCorrect(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;

    std::vector<data_t> dataVector(dataTable, dataTable + tableLen);
    startStopwatch(&timer);

    if (sortOrder == ORDER_ASC)
    {
        std::sort(dataVector.begin(), dataVector.end());
    }
    else
    {
        std::sort(dataVector.rbegin(), dataVector.rend());
    }

    double time = endStopwatch(timer);
    std::copy(dataVector.begin(), dataVector.end(), dataTable);

    return time;
}
