#include <stdio.h>
#include <stdlib.h>
#include <array>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


// TODO figure out, how to use only one compare function for parallel and sequential implementation.
int_t compareSeq(const void* elem1, const void* elem2) {
    return (((el_t*)elem1)->key - ((el_t*)elem2)->key);
}

void sortSequential(data_t* inputHost, data_t* outputHost, uint_t arrayLen, bool orderAsc) {
    // TODO
}

el_t* sortCorrect(el_t* input, uint_t tableLen) {
    el_t* output;
    LARGE_INTEGER timer;

    output = (el_t*)malloc(tableLen * sizeof(*output));
    std::copy(input, input + tableLen, output);

    startStopwatch(&timer);
    qsort(output, tableLen, sizeof(*output), compareSeq);
    endStopwatch(timer, "Executing C sort implementation");

    return output;
}
