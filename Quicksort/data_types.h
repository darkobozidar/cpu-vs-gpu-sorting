#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef uint32_t data_t;
typedef uint32_t uint_t;
typedef int32_t int_t;

typedef struct Element el_t;
typedef struct HostGlobalSequence h_glob_seq_t;
typedef struct DeviceGlobalSequence d_glob_seq_t;
typedef struct LocalSequence loc_seq_t;


/*
Enum used to denote, in which direction is the data transfered to during local and global quicksort.
*/
enum TransferDirection {
    PRIMARY_MEM_TO_BUFFER,
    BUFFER_TO_PRIMARY_MEM
};


/*
Key value pair used for sorting
*/
struct Element {
    data_t key;
    data_t val;
};

/*
Params for sequence used in GLOBAL quicksort on HOST.
Host needs different params for sequence beeing partitioned than device.
*/
struct HostGlobalSequence {
    uint_t start;
    uint_t length;
    uint_t oldStart;
    uint_t oldLength;
    data_t pivot;
    TransferDirection direction;

    void setInitSeq(uint_t tableLen, data_t initPivot);
    void setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
    void setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
};

/*
Params for sequence used in GLOBAL quicksort on DEVICE.
Device needs different params for sequence beeing partitioned than host.
*/
struct DeviceGlobalSequence {
    uint_t start;
    uint_t length;
    data_t pivot;
    TransferDirection direction;

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

    void setFromHostSeq(h_glob_seq_t globalSeqHost, uint_t threadBlocksPerSequence);
};

/*
Params for sequence used in LOCAL quicksort on DEVICE.
*/
struct LocalSequence {
    uint_t start;
    uint_t length;
    TransferDirection direction;

    void setLowerSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
    void setGreaterSeq(h_glob_seq_t globalSeqHost, d_glob_seq_t globalSeqDev);
    void setFromGlobalSeq(h_glob_seq_t globalParams);
};

#endif
