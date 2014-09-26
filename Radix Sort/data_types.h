#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;

// Key value pair used for sorting
struct Element {
    uint_t key;
    uint_t val;
};
typedef struct Element el_t;
struct Group {
    el_t el0;
    el_t el1;
    el_t el2;
    el_t el3;
};
typedef struct Group group_t;

#endif
