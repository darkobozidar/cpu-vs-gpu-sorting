#ifndef DATA_TYPES_COMMON_H
#define DATA_TYPES_COMMON_H

#include <stdint.h>

typedef uint32_t uint_t;
typedef int32_t int_t;

typedef uint32_t data_t;
typedef enum SortOrder order_t;

enum SortOrder {
    ORDER_ASC,
    ORDER_DESC
};

#endif
