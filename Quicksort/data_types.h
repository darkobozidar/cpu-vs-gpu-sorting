#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef uint32_t data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;

typedef struct Element el_t;
typedef struct HostGlobalSequence h_glob_seq_t;
typedef struct DeviceGlobalSequence d_glob_seq_t;
typedef struct LocalParams lparam_t;


// Key value pair used for sorting
struct Element {
    data_t key;
    data_t val;
};

// Params for sequence used in global quicksort on host.
struct HostGlobalSequence {
    uint_t start;
    uint_t length;
    uint_t oldStart;
    uint_t oldLength;
    data_t pivot;

    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;

    void setInitSequence(uint_t tableLen, data_t initPivot);
    void setLowerSequence(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
    void setGreaterSequence(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
};

// Params for sequence used in global quicksort on device.
struct DeviceGlobalSequence {
    uint_t start;
    uint_t length;
    data_t pivot;
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;

    // Every thread block in global quicksort kernel working on this sequence decreases this counter. This
    // way thread blocks know, which of them is last, so it can scatter pivots.
    uint_t blockCounter;

    // Counter used in global quicksort. Each thread block working on this sequence increments this counter
    // with number of elements lower/greater than pivot in it's corresponding data. This way every thread
    // block knows, on which offset it has to scatter it's corresponding data.
    uint_t offsetLower;
    uint_t offsetGreater;

    // Stores min/max values of every thread block.
    data_t minVal;
    data_t maxVal;

    void setSequence(h_glob_seq_t globalSeqHost, uint_t threadBlocksPerSequence);
};

struct LocalParams {
    uint_t start;
    uint_t length;
    // false: dataInput -> dataBuffer, true: dataBuffer -> dataInput
    bool direction;

    void lowerSequence(h_glob_seq_t oldParams, d_glob_seq_t deviceParams) {
        start = oldParams.oldStart;
        length = deviceParams.offsetLower;
        direction = !oldParams.direction;
    }

    void greaterSequence(h_glob_seq_t oldParams, d_glob_seq_t deviceParams) {
        start = oldParams.oldStart + oldParams.length - deviceParams.offsetGreater;
        length = deviceParams.offsetGreater;
        direction = !oldParams.direction;
    }

    void fromGlobalParams(h_glob_seq_t globalParams) {
        start = globalParams.start;
        length = globalParams.length;
        direction = globalParams.direction;
    }
};

#endif
