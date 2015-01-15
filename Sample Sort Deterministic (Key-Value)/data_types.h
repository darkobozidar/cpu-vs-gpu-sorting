#ifndef DATA_TYPES_H
#define DATA_TYPES_H


typedef enum SortOption sort_opt_t;

/*
Configures sort type - if only keys will be sorted or key-value pairs.
*/
enum SortOption
{
    SORT_KEYS_ONLY,
    SORT_KEYS_AND_VALUES
};

#endif
