#include <stdlib.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
Allocates host memory.
*/
void allocHostMemory(data_t **input, data_t **outputParallel, data_t **outputCorrect, uint_t tableLen)
{
    // Data input
    *input = (data_t*)malloc(tableLen * sizeof(**input));
    checkMallocError(*input);

    // Data output
    *outputParallel = (data_t*)malloc(tableLen * sizeof(**outputParallel));
    checkMallocError(*outputParallel);
    *outputCorrect = (data_t*)malloc(tableLen * sizeof(**outputCorrect));
    checkMallocError(*outputCorrect);
}

/*
Frees host memory.
*/
void freeHostMemory(data_t *input, data_t *outputParallel, data_t *outputCorrect) {
    free(input);
    free(outputParallel);
    free(outputCorrect);
}
