#include <stdio.h>
#include <stdlib.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


// TODO figure out, how to use only one compare function for parallel and sequential implementation.
int compareSeq(const void* elem1, const void* elem2)
{
    return *((data_t*)elem1) - *((data_t*)elem2);
}

void sortSequential(data_t* inputHost, data_t* outputHost, uint_t arrayLen, bool orderAsc)
{
    // TODO
}

double sortCorrect(data_t* input, data_t *output, uint_t tableLen)
{
    LARGE_INTEGER timer;

    startStopwatch(&timer);
    qsort(output, tableLen, sizeof(*output), compareSeq);
    return endStopwatch(timer);
}
