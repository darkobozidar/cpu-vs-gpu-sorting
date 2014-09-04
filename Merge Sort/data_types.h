#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;

// TODO comment
struct Element {
    uint_t key;
    uint_t val;
};
struct Sample {
    uint_t key;  // Has to be the same type as Element key
    uint_t rank;
};
typedef struct Element el_t;
typedef struct Sample sample_t;

#endif
