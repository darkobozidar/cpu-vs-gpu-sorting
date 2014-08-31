#include <stdio.h>
#include <stdlib.h>
#include <array>

#include "data_types.h"
#include "utils_host.h"
#include "sort_lib.h"


void sortSequential(data_t* inputHost, data_t* outputHost, uint_t arrayLen, bool orderAsc) {
    // TODO
}

data_t* sortCorrect(data_t* input, uint_t tableLen) {
    data_t* output;
    LARGE_INTEGER timer;

    output = (data_t*) malloc(tableLen * sizeof(*output));
    std::copy(input, input + tableLen, output);

    startStopwatch(&timer);
    //qsort(output, tableLen, sizeof(*output), compare);
    endStopwatch(timer, "Executing C sort implementation");

    return output;
}
