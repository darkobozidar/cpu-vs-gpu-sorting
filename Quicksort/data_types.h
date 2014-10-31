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
    data_t minVal;
    data_t maxVal;
    TransferDirection direction;

    void setInitSeq(uint_t tableLen, data_t initMinVal, data_t initMaxVal);
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

    // Holds the index of the first thread block assigned to this sequence. Multiple thread blocks can be
    // partitioning the same sequence, which length is not necessarily multiple of thread block length. It
    // is used to calculate the local thread block indexes for sequence and consequently which chunk of data
    // is assigned to current block.
    uint_t startThreadBlockIdx;
    // Holds the number of thread blocks assigned to this sequence. When thread blocks finish with execution
    // of global quicksort, they decrease this counter. This way eash thread block assigned to this sequence
    // knows, if it finnished last with the execution of kernel, so it can scatter pivots.
    uint_t threadBlockCounter;

    // Counter used in global quicksort. Each thread block working on this sequence increments this counter
    // with number of elements lower/greater than pivot in it's corresponding data. This way every thread
    // block knows, on which offset it has to scatter it's corresponding data.
    uint_t offsetLower;
    uint_t offsetGreater;

    // TODO comment
    data_t lowerSeqMaxVal;
    data_t greaterSeqMinVal;

    void setFromHostSeq(h_glob_seq_t globalSeqHost, uint_t startThreadBlock, uint_t threadBlocksPerSequence);
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
