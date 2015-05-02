#ifndef DATA_TYPES_COMMON_H
#define DATA_TYPES_COMMON_H

#include <stdint.h>


// Primitive data type definition
typedef uint32_t uint_t;
typedef int32_t int_t;

// Data type used for sorting.
// WARNING! When changing data type, update DATA_TYPE_BITS, MIN_VAL and MAX_VAL accordingly
typedef uint32_t data_t;
#define DATA_TYPE_BITS 32
#define MIN_VAL 0
#define MAX_VAL UINT32_MAX

// Determines sort order (ascending or descending)
typedef enum SortOrder order_t;
// Determines sort type (parallel, sequential or correct)
typedef enum SortType sort_type_t;
// Determines input distribution for random generator
typedef enum DataDistribution data_dist_t;

enum SortOrder
{
    ORDER_ASC,
    ORDER_DESC
};

enum SortType
{
    SORT_SEQUENTIAL_KEY_ONLY,
    SORT_SEQUENTIAL_KEY_VALUE,
    SORT_PARALLEL_KEY_ONLY,
    SORT_PARALLEL_KEY_VALUE
};

enum DataDistribution
{
    DISTRIBUTION_UNIFORM,
    DISTRIBUTION_GAUSSIAN,
    DISTRIBUTION_ZERO,
    DISTRIBUTION_BUCKET,
    DISTRIBUTION_STAGGERED,
    DISTRIBUTION_SORTED_ASC,
    DISTRIBUTION_SORTED_DESC
};

#endif
