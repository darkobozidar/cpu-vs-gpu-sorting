#ifndef STATISTICS_H
#define STATISTICS_H

// Keys only
void printTableSplitterKeysOnly();
void printTableHeaderKeysOnly();
void printTableLineKeysOnly(
    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential
);
void printStatisticsKeysOnly(double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly);

// Key-value
void printTableSplitterKeyValue();
void printTableHeaderKeyValue();
void printTableLineKeyValue(
    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential,
    bool isStableParallel, bool isStableSequential
);
void printStatisticsKeyValue(
    double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly, bool isStable
);

// Statistics utils
double getSpeedup(double **timers, sort_type_t sortType1, sort_type_t sortType2, uint_t testRepetitions);
bool isSortStable(data_t *keys, data_t *values, uint_t tableLen, order_t sortOrder);

#endif
