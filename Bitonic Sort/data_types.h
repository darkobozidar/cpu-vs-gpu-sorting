#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef uint32_t uint_t;
typedef int32_t int_t;

typedef uint32_t data_t;
typedef struct Element el_t;
typedef enum SortOrder order_t;

// Key value pair used for sorting
struct Element {
    data_t key;
    data_t val;
};

enum SortOrder {
    ORDER_ASC,
    ORDER_DESC
};

#endif
