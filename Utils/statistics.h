#ifndef STATISTICS_H
#define STATISTICS_H

void printTableSplitterKeysOnly();
void printTableHeaderKeysOnly(char *sortName);
void printTableLineKeysOnly(
    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential
);
void printStatisticsKeysOnly(double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly);

double getSpeedup(double **timers, sort_type_t sortType1, sort_type_t sortType2, uint_t testRepetitions);

#endif
