#include <stdio.h>
#include <stdint.h>
#include <random>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"


/*
Starts the stopwatch (remembers the current time).
*/
void startStopwatch(LARGE_INTEGER* start)
{
    QueryPerformanceCounter(start);
}

/*
Ends the stopwatch (calculates the difference between current time and parameter "start") and returns time
in miliseconds. Also prints out comment.
*/
double endStopwatch(LARGE_INTEGER start, char* comment)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER end;
    double elapsedTime;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&end);
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    if (comment != NULL)
    {
        printf("%s: %.5lf ms\n", comment, elapsedTime);
    }

    return elapsedTime;
}

/*
Ends the stopwatch (calculates the difference between current time and parameter "start") and returns time
in miliseconds.
*/
double endStopwatch(LARGE_INTEGER start)
{
    return endStopwatch(start, NULL);
}

/*
Keys are filled with random numbers and values are filled with consecutive naumbers.
- TODO use twister generator
*/
void fillTable(el_t *table, uint_t tableLen, uint_t interval)
{
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<uint_t> distribution(0, interval);

    for (uint_t i = 0; i < tableLen; i++) {
        table[i].key = distribution(generator);
        table[i].val = i;
    }
}

/*
Compares two arrays and prints out if they are the same or if they differ.
*/
void compareArrays(el_t* array1, el_t* array2, uint_t arrayLen)
{
    for (uint_t i = 0; i < arrayLen; i++)
    {
        if (array1[i].key != array2[i].key)
        {
            printf(
                "Arrays are different: array1[%d] = %d, array2[%d] = %d.\n",
                i, array1[i].key, i, array2[i].key
            );

            return;
        }
    }

    printf("Arrays are the same.\n");
}

/*
Prints out array from specified start to specified end index.
*/
void printTable(el_t *table, uint_t startIndex, uint_t endIndex)
{
    for (uint_t i = startIndex; i <= endIndex; i++)
    {
        char* separator = i == endIndex ? "" : ", ";
        printf("%2d%s", table[i].key, separator);
    }

    printf("\n\n");
}

/*
Prints out table from start to provided length.
*/
void printTable(el_t *table, uint_t tableLen)
{
    printTable(table, 0, tableLen - 1);
}

/*
Checks if there was an error.
*/
void checkMallocError(void *ptr)
{
    if (ptr == NULL)
    {
        printf("Error in host malloc\n.");
        getchar();
        exit(EXIT_FAILURE);
    }
}

/*
Return the next power of 2 for provided value. If value is already power of 2, it returns value.
*/
uint_t nextPowerOf2(uint_t value)
{
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;

    return value;
}
