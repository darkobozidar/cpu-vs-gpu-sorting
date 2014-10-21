#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;
// TODO create type for key

// Key value pair used for sorting
struct Element {
    uint_t key;
    uint_t val;
};
typedef struct Element el_t;

struct LocalParams {
    uint_t start;
    uint_t length;
};
typedef struct LocalParams lparam_t;

#endif
