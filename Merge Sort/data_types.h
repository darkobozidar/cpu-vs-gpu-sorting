#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

#include "../Utils/data_types_common.h"


typedef struct Sample sample_t;

// Key value pair used for sorting
struct Sample
{
    data_t key;
    uint_t val;
};

#endif
