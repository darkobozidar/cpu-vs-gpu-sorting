#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>


typedef struct Interval interval_t;

/*
Holds 2 intervals needed for IBR bitonic sort.
Intervals are represented with offset (index) in array and with length.
*/
struct Interval
{
    uint32_t offset0;
    uint32_t length0;
    uint32_t offset1;
    uint32_t length1;
};

#endif
