#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>


// TODO comment
typedef struct Interval interval_t;

struct Interval {
    uint32_t offset0;
    uint32_t length0;
    uint32_t offset1;
    uint32_t length1;
};

#endif
