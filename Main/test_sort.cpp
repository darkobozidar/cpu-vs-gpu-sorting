#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "../Utils/host.h"
#include "../Utils/file.h"
#include "../Utils/generator.h"
#include "../Utils/sort_correct.h"
#include "constants.h"


/*
Prints line in statistics table.
*/
void printTableLine()
{
    printf("==================================================================\n");
}

/*
Prints statistics table header.
*/
void printTableHeader()
{
    printTableLine();
    printf("|| #     ||      TIME     |    SORT RATE   || CORRECT || STABLE ||\n");
    printTableLine();
}

/*
Prints sort statistics.
*/
void printSortStatistics(uint_t iteration, double time, uint_t arrayLength, int_t isCorrect, int_t isStable)
{
    char *isCorrectOutput = isCorrect ? "YES" : "NO";
    char *isStableOutput = isStable == -1 ? "/" : (isStable == 1 ? "YES" : "NO");

    printf(
        "|| %5d || %10.2lf ms | %10.2lf M/s ||    %3s  ||   %3s  ||\n", iteration + 1, time,
        arrayLength / 1000.0 / time, isCorrectOutput, isStableOutput
    );
}

/*
Folder path to specified distribution.
*/
std::string folderPathDistribution(data_dist_t distribution)
{
    std::string distFolderName(FOLDER_SORT_TIMERS);
    distFolderName += strCapitalize(getDistributionName(distribution));
    return distFolderName + "/";
}

/*
Creates the folder structure in order to save sort statistics to disc.
*/
void createFolderStructure(std::vector<data_dist_t> distributions)
{
    createFolder(FOLDER_SORT_ROOT);
    createFolder(FOLDER_SORT_TIMERS);
    createFolder(FOLDER_SORT_CORRECTNESS);
    createFolder(FOLDER_SORT_STABILITY);
    createFolder(FOLDER_SORT_CORRECTNESS FOLDER_LOG);
    createFolder(FOLDER_SORT_STABILITY FOLDER_LOG);

    // Creates a folder for every distribution, inside which creates a folder for data type.
    for (std::vector<data_dist_t>::iterator dist = distributions.begin(); dist != distributions.end(); dist++)
    {
        createFolder(folderPathDistribution(*dist));
    }
}

/*
Generates file name of unsorted array.
*/
std::string fileNameUnsortedArr(uint_t iteration)
{
    return std::string(FILE_UNSORTED_ARRAY) + "_" + std::to_string(iteration) + FILE_EXTENSION;
}

/*
Generates file name of sorted array.
*/
std::string fileNameSortedArr(uint_t iteration)
{
    return std::string(FILE_SORTED_ARRAY) + '_' + std::to_string(iteration) + FILE_EXTENSION;
}

/*
Checks if sorted values are unique.
Not so trivial in some sorts - for example quicksort.
*/
void checkValuesUniqueness(data_t *values, uint_t arrayLength)
{
    uint_t *uniquenessTable = (uint_t*)malloc(arrayLength * sizeof(*uniquenessTable));

    for (uint_t i = 0; i < arrayLength; i++)
    {
        uniquenessTable[i] = 0;
    }

    for (uint_t i = 0; i < arrayLength; i++)
    {
        if (values[i] < 0 || values[i] > arrayLength - 1)
        {
            printf("Value out of range: %d\n", values[i]);
            getchar();
            exit(EXIT_FAILURE);
        }
        else if (++uniquenessTable[values[i]] > 1)
        {
            printf("Duplicate value: %d\n", values[i]);
            getchar();
            exit(EXIT_FAILURE);
        }
    }

    free(uniquenessTable);
}

/*
Determines if array is sorted in stable manner.
*/
bool isSortStable(data_t *keys, data_t *values, uint_t arrayLength)
{
    if (arrayLength == 1)
    {
        return true;
    }

    // Generally not needed
    // checkValuesUniqueness(values, arrayLength);

    for (uint_t i = 1; i < arrayLength; i++)
    {
        // For same keys, values have to be ordered in ascending order.
        if (keys[i - 1] == keys[i] && values[i - 1] > values[i])
        {
            return false;
        }
    }

    return true;
}

/*
Writes the time to file
*/
void writeTimeToFile(
    SortSequential *sort, data_dist_t distribution, double time, bool sortingKeyOnly, bool isLastTestRepetition
)
{
    std::string folderDistribution = folderPathDistribution(distribution);
    std::string filePath = folderDistribution + strSlugify(sort->getSortName(sortingKeyOnly)) + FILE_EXTENSION;
    std::fstream file;
    file.open(filePath, std::fstream::app);

    file << time;
    file << (isLastTestRepetition ? FILE_NEW_LINE_CHAR : FILE_SEPARATOR_CHAR);

    file.close();
}

/*
Writes bolean to a file. Needed to write sort correctness and sort stability.
*/
void writeBoleanToFile(
    std::string folderName, bool val, SortSequential *sort, data_dist_t distribution, uint_t arrayLength,
    order_t sortOrder, bool sortingKeyOnly
)
{
    std::string filePath = folderName + strSlugify(sort->getSortName(sortingKeyOnly)) + FILE_EXTENSION;
    std::fstream file;

    // Outputs boolean
    file.open(filePath, std::fstream::app);
    file << val << FILE_SEPARATOR_CHAR;
    file.close();

    // Prints log in case if boolean is false
    if (!val)
    {
        std::string fileLog = folderName + FOLDER_LOG;
        fileLog += strSlugify(sort->getSortName(sortingKeyOnly)) + FILE_EXTENSION;

        file.open(fileLog, std::fstream::app);
        file << getDistributionName(distribution) << " ";
        file << arrayLength << " ";
        file << (sortOrder == ORDER_ASC ? "ASC" : "DESC");
        file << FILE_NEW_LINE_CHAR;
        file.close();
    }
}

/*
Times sort with stopwatch, checks if sort is stable and checks if sort is ordering data correctly, than saves
this statistics to file.
*/
void testSort(
    SortSequential *sort, data_dist_t distribution, data_t *keys, data_t *keysCopy, data_t *values,
    uint_t arrayLength, order_t sortOrder, uint_t interval, uint_t iteration, uint_t testRepetitions,
    bool sortingKeyOnly
)
{
    // For every sort array is filled with new random values. It would be better if array was generated only
    // once and saved to file, but this works much slower.
    fillArrayKeyOnly(keys, arrayLength, interval, distribution);
    std::copy(keys, keys + arrayLength, keysCopy);

    if (sortingKeyOnly)
    {
        sort->sort(keys, arrayLength, sortOrder);
    }
    else
    {
        fillArrayValueOnly(values, arrayLength);
        sort->sort(keys, values, arrayLength, sortOrder);
    }

    double time = sort->getSortTime();
    writeTimeToFile(sort, distribution, time, sortingKeyOnly, iteration == testRepetitions - 1);

    // Sort correctness test
    if (!(distribution == DISTRIBUTION_SORTED_ASC && sortOrder == ORDER_ASC ||
          distribution == DISTRIBUTION_SORTED_DESC && sortOrder == ORDER_DESC)
    )
    {
        sortCorrect(keysCopy, arrayLength, sortOrder);
    }

    bool isCorrect = compareArrays(keys, keysCopy, arrayLength);
    writeBoleanToFile(
        FOLDER_SORT_CORRECTNESS, isCorrect, sort, distribution, arrayLength, sortOrder, sortingKeyOnly
    );

    // Key-value sort has to be tested for stability
    int_t isStable = -1;
    if (!sortingKeyOnly)
    {
        isStable = isSortStable(keys, values, arrayLength);
        writeBoleanToFile(
            FOLDER_SORT_STABILITY, isStable, sort, distribution, arrayLength, sortOrder, sortingKeyOnly
        );
    }

    printSortStatistics(iteration, time, arrayLength, isCorrect, isStable);
}

/*
Tests the sort and generates results.
*/
void generateSortTestResults(
    SortSequential *sort, data_dist_t distribution, data_t *keys, data_t *keysCopy, data_t *values,
    uint_t arrayLength, order_t sortOrder, uint_t interval, uint_t testRepetitions, bool sortingKeyOnly
)
{
    printf("> Distribution: %s\n", getDistributionName(distribution));
    printf("> Data type: %s\n", typeid(data_t).name());
    printf("> Array length: %d\n", arrayLength);
    printf("> %s\n", sort->getSortName(sortingKeyOnly).c_str());
    printTableHeader();

    // Tests sort for key only
    for (uint_t iter = 0; iter < testRepetitions; iter++)
    {
        testSort(
            sort, distribution, keys, keysCopy, values, arrayLength, sortOrder, interval, iter, testRepetitions,
            sortingKeyOnly
        );
    }

    printTableLine();
}

/*
Tests all provided sorts for all provided distributions.
*/
void generateStatistics(
    std::vector<SortSequential*> sorts, std::vector<data_dist_t> distributions, uint_t arrayLength,
    order_t sortOrder, uint_t testRepetitions, uint_t interval
)
{
    createFolderStructure(distributions);
    std::string arrayLenStr = std::to_string(arrayLength) + std::string(FILE_NEW_LINE_CHAR);
    appendToFile(FILE_ARRAY_LENGTHS, arrayLenStr);

    data_t *keys = (data_t*)malloc(arrayLength * sizeof(*keys));
    checkMallocError(keys);
    data_t *keysCopy = (data_t*)malloc(arrayLength * sizeof(*keysCopy));
    checkMallocError(keysCopy);
    data_t *values = (data_t*)malloc(arrayLength * sizeof(*values));
    checkMallocError(values);

    for (std::vector<SortSequential*>::iterator sort = sorts.begin(); sort != sorts.end(); sort++)
    {
        for (std::vector<data_dist_t>::iterator dist = distributions.begin(); dist != distributions.end(); dist++)
        {
            // Sort key-only
            generateSortTestResults(
                *sort, *dist, keys, keysCopy, values, arrayLength, sortOrder, interval, testRepetitions, true
            );

            printf("\n\n");

            // Sort key-value pairs
            generateSortTestResults(
                *sort, *dist, keys, keysCopy, values, arrayLength, sortOrder, interval, testRepetitions, false
            );

            printf("\n\n");
        }

        (*sort)->memoryDestroy();
    }

    free(keys);
    free(keysCopy);
    free(values);
}
