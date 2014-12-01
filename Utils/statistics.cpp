#include <stdio.h>
#include <math.h>

#include "data_types_common.h"


/*
Prints out table vertical splitter if only keys are sorted.
*/
void printTableSplitterKeysOnly()
{
    printf("====================================================================================================\n");
}

/*
Prints out table header if only keys are sorted.
*/
void printTableHeaderKeysOnly()
{
    printTableSplitterKeysOnly();
    printf("||     # ||             PARALLEL              ||             SEQUENTIAL            ||   CORRECT   ||\n");
    printTableSplitterKeysOnly();
    printf("||     # ||     time    |      rate    |  ok  ||     time    |      rate    |  ok  ||     time    ||\n");
    printTableSplitterKeysOnly();
}

/*
Prints out table line with data if only keys are sorted.
*/
void printTableLineKeysOnly(
    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential
)
{
    printf(
        "|| %5d || %8.2lf ms | %8.2lf M/s | %s  || %8.2lf ms | %8.2lf M/s | %s  || %8.2lf ms ||\n", iter + 1,
        timers[SORT_PARALLEL][iter], tableLen / 1000.0 / timers[SORT_PARALLEL][iter],
        areEqualParallel ? "YES" : " NO",
        timers[SORT_SEQUENTIAL][iter], tableLen / 1000.0 / timers[SORT_SEQUENTIAL][iter],
        areEqualSequential ? "YES" : " NO",
        timers[SORT_CORRECT][iter]
    );
}

/*
Prints out table vertical splitter if key-value pairs are sorted.
*/
void printTableSplitterKeyValue()
{
    printf("======================================================================================================================\n");
}

/*
Prints out table header if key-value pairs are sorted.
*/
void printTableHeaderKeyValue()
{
    printTableSplitterKeyValue();
    printf("||     # ||                  PARALLEL                  ||                  SEQUENTIAL                ||   CORRECT   ||\n");
    printTableSplitterKeyValue();
    printf("||     # ||     time    |      rate    |  ok  | stable ||     time    |      rate    |  ok  | stable ||     time    ||\n");
    printTableSplitterKeyValue();
}

/*
Prints out table line with data if key-value pairs are sorted.
*/
void printTableLineKeyValue(
    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential,
    bool isStableParallel, bool isStableSequential
)
{
    printf(
        "|| %5d || %8.2lf ms | %8.2lf M/s | %s  |   %s  || %8.2lf ms | %8.2lf M/s | %s  |   %s  || %8.2lf ms ||\n",
        iter + 1,
        timers[SORT_PARALLEL][iter], tableLen / 1000.0 / timers[SORT_PARALLEL][iter],
        areEqualParallel ? "YES" : " NO", isStableParallel ? "YES" : " NO",
        timers[SORT_SEQUENTIAL][iter], tableLen / 1000.0 / timers[SORT_SEQUENTIAL][iter],
        areEqualSequential ? "YES" : " NO", isStableSequential ? "YES" : " NO",
        timers[SORT_CORRECT][iter]
    );
}

/*
Prints out statistics of sort if only keys are sorted.
*/
void printStatisticsKeysOnly(double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly)
{
    double timeSum = 0;

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        timeSum += timers[i];
    }

    double avgTime = timeSum / (double)testRepetitions;
    timeSum = 0;

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        timeSum += pow(avgTime - timers[i], 2);
    }

    double deviationTime = sqrt(timeSum);

    printf("Average sort time:  %8.2lf ms\n", avgTime);
    printf("Average sort rate:  %8.2lf M/s\n", tableLen / 1000.0 / avgTime);
    printf("Standard deviation: %8.2lf ms  (%.2lf%%)\n", deviationTime, deviationTime / avgTime * 100);
    printf("Sorting correctly:  %8s\n", sortsCorrectly ? "YES" : "NO");
}

void printStatisticsKeyValue(
    double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly, bool isStable
)
{
    printStatisticsKeysOnly(timers, testRepetitions, tableLen, sortsCorrectly);
    printf("Is sort stable:     %8s\n", isStable ? "YES" : "NO");
}

/*
Returns a speedup for 2 provided sort types.
*/
double getSpeedup(double **timers, sort_type_t sortType1, sort_type_t sortType2, uint_t testRepetitions)
{
    double totalTimeSortType1 = 0;
    double totalTimeSortType2 = 0;

    for (uint_t i = 0; i < testRepetitions; i++)
    {
        totalTimeSortType1 += timers[sortType1][i];
        totalTimeSortType2 += timers[sortType2][i];
    }

    return totalTimeSortType1 / totalTimeSortType2;
}

/*
Determines if array is sorted in stable manner.
// TODO order DESC
*/
bool isSortStable(data_t *keys, data_t *values, uint_t tableLen)
{
    if (tableLen == 1)
    {
        return true;
    }

    for (uint_t i = 1; i < tableLen; i++)
    {
        // For same keys, values have to be ordered in ascending order.
        if (keys[i - 1] == keys[i] && values[i - 1] > values[i])
        {
            return false;
        }
    }

    return true;
}
