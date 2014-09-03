#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;

// TODO comment
struct Element {
    uint32_t key;
    uint32_t val;
};
struct RankElement {
    struct Element el;
    uint_t rank;
};
typedef struct Element el_t;
typedef struct RankElement rank_el_t;

#endif
