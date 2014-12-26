#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

#include "../Utils/data_types_common.h"


typedef struct Sample sample_t;

// Key value pair used for sorting
struct Sample
{
    // Sample value
    data_t value;
    // Sample rank in current sorted block. Needed when samples per merged block are sorted in order to memorize
    // the original position of sample in sorted block.
    uint_t index;
};

#endif
