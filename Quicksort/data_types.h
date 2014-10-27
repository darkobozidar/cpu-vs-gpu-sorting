#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;
// TODO create type for key

typedef struct Element el_t;
typedef struct HostGlobalParams h_gparam_t;
typedef struct DeviceGlobalParams d_gparam_t;
typedef struct LocalParams lparam_t;

// Key value pair used for sorting
struct Element {
    uint_t key;
    uint_t val;
};

// Params for each sub-sequence used in global quicksort on host.
struct HostGlobalParams {
    uint_t start;
    uint_t length;
    uint_t oldStart;
    uint_t oldLength;

    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;
    // TODO use correct data type
    uint_t pivot;

    void setDefaultParams(uint_t tableLen) {
        start = 0;
        length = tableLen;
        oldStart = start;
        oldLength = length;
        direction = false;
    }

    void lowerSequence(h_gparam_t oldParams, d_gparam_t deviceParams);
    void greaterSequence(h_gparam_t oldParams, d_gparam_t deviceParams);
};

// Params for each sub-sequence used in global quicksort on device.
struct DeviceGlobalParams {
    uint_t start;
    uint_t length;
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;
    // TODO use correct data type
    uint_t pivot;
    uint_t blockCounter;

    uint_t offsetLower;
    uint_t offsetGreater;

    // Can't use el_t because of avg(), which is used for pivot calculation
    // TODO use correct data type
    uint_t minVal;
    uint_t maxVal;

    void fromHostGlobalParams(h_gparam_t hostGlobalParams, uint_t threadBlocksPerSequence) {
        start = hostGlobalParams.start;
        length = hostGlobalParams.length;
        direction = hostGlobalParams.direction;
        pivot = hostGlobalParams.pivot;
        blockCounter = threadBlocksPerSequence;

        offsetLower = 0;
        offsetGreater = 0;

        minVal = UINT32_MAX;
        maxVal = 0;
    }
};

struct LocalParams {
    uint_t start;
    uint_t length;
    // TODO enum
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;

    void lowerSequence(struct HostGlobalParams oldParams, struct DeviceGlobalParams deviceParams) {
        start = oldParams.oldStart;
        length = deviceParams.offsetLower;
        direction = !oldParams.direction;
    }

    void greaterSequence(struct HostGlobalParams oldParams, struct DeviceGlobalParams deviceParams) {
        start = oldParams.oldStart + oldParams.length - deviceParams.offsetGreater;
        length = deviceParams.offsetGreater;
        direction = !oldParams.direction;
    }

    void fromGlobalParams(struct HostGlobalParams globalParams) {
        start = globalParams.start;
        length = globalParams.length;
        direction = globalParams.direction;
    }
};

#endif
