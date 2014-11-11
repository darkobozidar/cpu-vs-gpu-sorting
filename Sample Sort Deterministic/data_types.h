#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef uint32_t uint_t;
typedef int32_t int_t;

// Data type used for input elements
typedef uint32_t data_t;
typedef struct Element el_t;

/*
Key value pair used for sorting
*/
struct Element {
    data_t key;
    data_t val;
};

#endif
