#ifndef DATA_TYPES_COMMON_H
#define DATA_TYPES_COMMON_H

#include <stdint.h>

// Primitive data type definition
typedef uint32_t uint_t;
typedef int32_t int_t;

// Data type used for sorting
typedef uint32_t data_t;
// Determines sort order (ascending or descending)
typedef enum SortOrder order_t;
// Determines sort type (parallel, sequential or correct)
typedef enum SortType sort_type_t;

enum SortOrder {
    ORDER_ASC,
    ORDER_DESC
};

enum SortType {
    SORT_PARALLEL,
    SORT_SEQUENTIAL,
    SORT_CORRECT
};

#endif
