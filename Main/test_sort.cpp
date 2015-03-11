#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "../Utils/host.h"
#include "../Utils/file.h"
#include "../Utils/generator.h"
#include "constants.h"


std::string folderPathDistribution(data_dist_t distribution)
{
    std::string distFolderName(FOLDER_SORT_TIMERS);
    distFolderName += strCapitalize(getDistributionName(distribution));
    return distFolderName;
}

std::string folderPathDataType(data_dist_t distribution)
{
    std::string distFolderName = folderPathDistribution(distribution);
    return distFolderName + "/" + strReplace(typeid(data_t).name(), ' ', '_');
}

void createFolderStructure(std::vector<data_dist_t> distributions)
{
    createFolder(FOLDER_SORT_ROOT);
    createFolder(FOLDER_SORT_TEMP);
    createFolder(FOLDER_SORT_TIMERS);

    // Creates a folder for every distribution, inside which creates a folder for data type.
    for (std::vector<data_dist_t>::iterator dist = distributions.begin(); dist != distributions.end(); dist++)
    {
        createFolder(folderPathDistribution(*dist));
        createFolder(folderPathDataType(*dist));
    }
}

void writeSortTimeToFile(Sort *sort, data_dist_t distribution, double time, bool isLastTest)
{
    std::string filePath = folderPathDataType(distribution) + "/" + strReplace(sort->getSortName(), ' ', '_') + ".txt";
    std::fstream file;
    file.open(filePath, std::fstream::app);

    file << time;
    file << (isLastTest ? "\n" : "\t");

    file.close();
}

/*
Allocates memory, performs sorts and times it with stopwatch, destroys extra memory and returns time.
*/
double stopwatchSort(
    Sort *sort, data_dist_t distribution, data_t *keys, data_t *values, uint_t arrayLength, order_t sortOrder
)
{
    readArrayFromFile(FILE_UNSORTED_ARRAY, keys, arrayLength);

    // Allocates memory key only sort OR key-value sort
    if (sort->getSortType() == SORT_SEQUENTIAL_KEY_ONLY || sort->getSortType() == SORT_PARALLEL_KEY_ONLY)
    {
        sort->memoryAllocate(keys, arrayLength);
    }
    else
    {
        sort->memoryAllocate(keys, values, arrayLength);
    }

    // Memory data transfer for parallel sorts
    if (sort->getSortType() == SORT_PARALLEL_KEY_ONLY)
    {
        sort->memoryCopyHostToDevice(keys, arrayLength);
        cudaDeviceSynchronize();
    }
    else if (sort->getSortType() == SORT_PARALLEL_KEY_VALUE)
    {
        fillArrayValueOnly(values, arrayLength);
        sort->memoryCopyHostToDevice(keys, values, arrayLength);
        cudaDeviceSynchronize();
    }

    LARGE_INTEGER timer;
    startStopwatch(&timer);

    // Sort depending if it sorts only keys or key-value pairs
    if (sort->getSortType() == SORT_SEQUENTIAL_KEY_ONLY || sort->getSortType() == SORT_PARALLEL_KEY_ONLY)
    {
        sort->sort(keys, arrayLength, sortOrder);
    }
    else
    {
        sort->sort(keys, values, arrayLength, sortOrder);
    }

    // Waits for device to finish
    if (sort->getSortType() == SORT_PARALLEL_KEY_ONLY || sort->getSortType() == SORT_PARALLEL_KEY_VALUE)
    {
        cudaError_t error = cudaDeviceSynchronize();
        checkCudaError(error);
    }

    double time = endStopwatch(timer);
    sort->memoryDestroy();

    return time;
}

/*
Tests all provided sorts for all provided distributions.
*/
void testSorts(
    std::vector<Sort*> sorts, std::vector<data_dist_t> distributions, uint_t arrayLenStart, uint_t arrayLenEnd,
    order_t sortOrder, uint_t testRepetitions, uint_t interval
)
{
    createFolderStructure(distributions);

    for (std::vector<data_dist_t>::iterator dist = distributions.begin(); dist != distributions.end(); dist++)
    {
        for (uint_t arrayLength = arrayLenStart; arrayLength <= arrayLenEnd; arrayLength *= 2)
        {
            data_t *keys = (data_t*)malloc(arrayLength * sizeof(*keys));
            checkMallocError(keys);
            data_t *values = (data_t*)malloc(arrayLength * sizeof(*values));
            checkMallocError(values);

            for (uint_t iter = 0; iter < testRepetitions; iter++)
            {
                printf("> Test iteration: %d\n", iter);
                printf("> Distribution: %s\n", getDistributionName(*dist));
                printf("> Array length: %d\n", arrayLength);
                printf("> Data type: %s\n", typeid(data_t).name());

                // All the sort algorithms have to sort the same array
                fillArrayKeyOnly(keys, arrayLength, interval, *dist);
                saveArrayToFile(FILE_UNSORTED_ARRAY, keys, arrayLength);

                for (std::vector<Sort*>::iterator sort = sorts.begin(); sort != sorts.end(); sort++)
                {
                    double time = stopwatchSort(*sort, *dist, keys, values, arrayLength, sortOrder);
                    writeSortTimeToFile(*sort, *dist, time, iter == testRepetitions - 1);
                }

                printf("\n");
            }

            free(keys);
            free(values);
        }

        printf("\n\n");
    }
}


///*
//Prints out table vertical splitter if only keys are sorted.
//*/
//void printTableSplitterKeyOnly()
//{
//    printf("===================================================================================================================\n");
//}
//
///*
//Prints out table header if only keys are sorted.
//*/
//void printTableHeaderKeysOnly()
//{
//    printTableSplitterKeyOnly();
//    printf("||     # ||             PARALLEL              ||             SEQUENTIAL            ||            CORRECT         ||\n");
//    printTableSplitterKeyOnly();
//    printf("||     # ||     time    |      rate    |  ok  ||     time    |      rate    |  ok  ||     time    |      rate    ||\n");
//    printTableSplitterKeyOnly();
//}
//
///*
//Prints out table line with data if only keys are sorted.
//*/
//void printTableLineKeysOnly()
//{
//
//}

///*
//Prints out table vertical splitter if key-value pairs are sorted.
//*/
//void printTableSplitterKeyValue()
//{
//    printf("=====================================================================================================================================\n");
//}
//
///*
//Prints out table header if key-value pairs are sorted.
//*/
//void printTableHeaderKeyValue()
//{
//    printTableSplitterKeyValue();
//    printf("||     # ||                  PARALLEL                  ||                  SEQUENTIAL                ||            CORRECT         ||\n");
//    printTableSplitterKeyValue();
//    printf("||     # ||     time    |      rate    |  ok  | stable ||     time    |      rate    |  ok  | stable ||     time    |      rate    ||\n");
//    printTableSplitterKeyValue();
//}
//
///*
//Prints out table line with data if key-value pairs are sorted.
//*/
//void printTableLineKeyValue(
//    double **timers, uint_t iter, uint_t tableLen, bool areEqualParallel, bool areEqualSequential,
//    bool isStableParallel, bool isStableSequential
//)
//{
//    printf(
//        "|| %5d || %8.2lf ms | %8.2lf M/s | %s  |   %s  || %8.2lf ms | %8.2lf M/s | %s  |   %s  || %8.2lf ms | %8.2lf M/s ||\n",
//        iter + 1,
//        timers[SORT_PARALLEL][iter], tableLen / 1000.0 / timers[SORT_PARALLEL][iter],
//        areEqualParallel ? "YES" : " NO", isStableParallel ? "YES" : " NO",
//        timers[SORT_SEQUENTIAL][iter], tableLen / 1000.0 / timers[SORT_SEQUENTIAL][iter],
//        areEqualSequential ? "YES" : " NO", isStableSequential ? "YES" : " NO",
//        timers[SORT_CORRECT][iter], tableLen / 1000.0 / timers[SORT_CORRECT][iter]
//    );
//}
//
///*
//Prints out statistics of sort if only keys are sorted.
//*/
//void printStatisticsKeysOnly(double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly)
//{
//    double timeSum = 0;
//
//    for (uint_t i = 0; i < testRepetitions; i++)
//    {
//        timeSum += timers[i];
//    }
//
//    double avgTime = timeSum / (double)testRepetitions;
//    timeSum = 0;
//
//    for (uint_t i = 0; i < testRepetitions; i++)
//    {
//        timeSum += pow(avgTime - timers[i], 2);
//    }
//
//    double deviationTime = sqrt(timeSum);
//
//    printf("Average sort time:  %8.2lf ms\n", avgTime);
//    printf("Average sort rate:  %8.2lf M/s\n", tableLen / 1000.0 / avgTime);
//    printf("Standard deviation: %8.2lf ms  (%.2lf%%)\n", deviationTime, deviationTime / avgTime * 100);
//    printf("Sorting correctly:  %8s\n", sortsCorrectly ? "YES" : "NO");
//}
//
//void printStatisticsKeyValue(
//    double *timers, uint_t testRepetitions, uint_t tableLen, bool sortsCorrectly, bool isStable
//)
//{
//    printStatisticsKeysOnly(timers, testRepetitions, tableLen, sortsCorrectly);
//    printf("Is sort stable:     %8s\n", isStable ? "YES" : "NO");
//}
//
///*
//Returns a speedup for 2 provided sort types.
//*/
//double getSpeedup(double **timers, sort_type_t sortType1, sort_type_t sortType2, uint_t testRepetitions)
//{
//    double totalTimeSortType1 = 0;
//    double totalTimeSortType2 = 0;
//
//    for (uint_t i = 0; i < testRepetitions; i++)
//    {
//        totalTimeSortType1 += timers[sortType1][i];
//        totalTimeSortType2 += timers[sortType2][i];
//    }
//
//    return totalTimeSortType1 / totalTimeSortType2;
//}
//
///*
//Checks if sorted values are unique.
//Not so trivial in some sorts - for example quicksort.
//*/
//void checkValuesUniqueness(data_t *values, uint_t tableLen)
//{
//    uint_t *uniquenessTable = (uint_t*)malloc(tableLen * sizeof(*uniquenessTable));
//
//    for (uint_t i = 0; i < tableLen; i++)
//    {
//        uniquenessTable[i] = 0;
//    }
//
//    for (uint_t i = 0; i < tableLen; i++)
//    {
//        if (values[i] < 0 || values[i] > tableLen - 1)
//        {
//            printf("Value out of range: %d\n", values[i]);
//            getchar();
//            exit(EXIT_FAILURE);
//        }
//        else if (++uniquenessTable[values[i]] > 1)
//        {
//            printf("Duplicate value: %d\n", values[i]);
//            getchar();
//            exit(EXIT_FAILURE);
//        }
//    }
//
//    free(uniquenessTable);
//}
//
///*
//Determines if array is sorted in stable manner.
//*/
//bool isSortStable(data_t *keys, data_t *values, uint_t tableLen)
//{
//    if (tableLen == 1)
//    {
//        return true;
//    }
//
//    // Generally not needed
//    // checkValuesUniqueness(values, tableLen);
//
//    for (uint_t i = 1; i < tableLen; i++)
//    {
//        // For same keys, values have to be ordered in ascending order.
//        if (keys[i - 1] == keys[i] && values[i - 1] > values[i])
//        {
//            return false;
//        }
//    }
//
//    return true;
//}
