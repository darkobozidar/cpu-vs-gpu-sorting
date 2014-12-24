#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

#include "../Utils/data_types_common.h"


// Key value pair used for sorting
struct Element {
    uint_t key;
    uint_t val;
};
typedef struct Element el_t;

#endif
