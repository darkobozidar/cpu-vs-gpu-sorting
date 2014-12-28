#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    data_t **h_minMaxValues, h_glob_seq_t **globalSeqHost, h_glob_seq_t **globalSeqHostBuffer,
    d_glob_seq_t **globalSeqDev, uint_t **globalSeqIndexes, loc_seq_t **localSeq, double ***timers,
    uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect,
    data_t *h_minMaxValues, h_glob_seq_t *globalSeqHost, h_glob_seq_t *globalSeqHostBuffer,
    d_glob_seq_t *globalSeqDev, uint_t *globalSeqIndexes, loc_seq_t *localSeq, double **timers
);

void allocDeviceMemory(
    data_t **dataTable, data_t **dataBuffer, d_glob_seq_t **globalSeqDev, uint_t **globalSeqIndexes,
    loc_seq_t **localSeq, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, d_glob_seq_t *globalSeqDev, uint_t *globalSeqIndexes, loc_seq_t *localSeq
);

#endif
