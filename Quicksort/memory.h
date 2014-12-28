#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include "../Utils/data_types_common.h"


void allocHostMemory(
    data_t **input, data_t **outputParallel, data_t **outputSequential, data_t **outputCorrect,
    h_glob_seq_t **h_globalSeqHost, h_glob_seq_t **h_globalSeqHostBuffer, d_glob_seq_t **h_globalSeqDev,
    uint_t **h_globalSeqIndexes, loc_seq_t **h_localSeq, double ***timers, uint_t tableLen, uint_t testRepetitions
);
void freeHostMemory(
    data_t *input, data_t *outputParallel, data_t *outputSequential, data_t *outputCorrect,
    h_glob_seq_t *h_globalSeqHost, h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev,
    uint_t *h_globalSeqIndexes, loc_seq_t *h_localSeq, double **timers
);

void allocDeviceMemory(
    data_t **dataTable, data_t **dataBuffer, d_glob_seq_t **globalSeqDev, uint_t **globalSeqIndexes,
    loc_seq_t **localSeq, uint_t tableLen
);
void freeDeviceMemory(
    data_t *dataTable, data_t *dataBuffer, d_glob_seq_t *globalSeqDev, uint_t *globalSeqIndexes, loc_seq_t *localSeq
);

#endif
