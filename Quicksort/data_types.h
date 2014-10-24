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

// Params for each sub-sequence used in global quicksort on host.
struct HostGlobalParams {
    uint_t start;
    uint_t length;
    uint_t oldStart;
    uint_t oldLength;

    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;
    el_t pivot;
};
typedef struct HostGlobalParams h_gparam_t;

// Params for each sub-sequence used in global quicksort on device.
struct DeviceGlobalParams {
    uint_t start;
    uint_t length;
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;

    uint_t offsetLower;
    uint_t offsetGreater;

    // Can't use el_t because of avg(), which is used for pivot calculation
    // TODO use correct data type
    uint_t minValLower;
    uint_t maxValLower;
    uint_t minValGreater;
    uint_t maxValGreater;
};
typedef struct DeviceGlobalParams d_gparam_t;

struct LocalParams {
    uint_t start;
    uint_t length;
    // TODO enum
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;
};
typedef struct LocalParams lparam_t;

#endif
