#include <stdio.h>
#include <stdint.h>
#include <string.h>
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
in milliseconds. Also prints out comment.
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
in milliseconds.
*/
double endStopwatch(LARGE_INTEGER start)
{
    return endStopwatch(start, NULL);
}

/*
Compares two arrays and prints out if they are the same or if they differ.
*/
bool compareArrays(data_t* array1, data_t* array2, uint_t arrayLen)
{
    for (uint_t i = 0; i < arrayLen; i++)
    {
        if (array1[i] != array2[i])
        {
            return false;
        }
    }

    return true;
}

/*
Prints out array from specified start to specified end index.
*/
void printTable(data_t *table, uint_t startIndex, uint_t endIndex)
{
    for (uint_t i = startIndex; i <= endIndex; i++)
    {
        char* separator = i == endIndex ? "" : ", ";
        printf("%2d%s", table[i], separator);
    }

    printf("\n\n");
}

/*
Prints out table from start to provided length.
*/
void printTable(data_t *table, uint_t tableLen)
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
        printf("Error in host malloc.\n");
        exit(EXIT_FAILURE);
    }
}

/*
Tests if number is power of 2.
*/
bool isPowerOfTwo(uint_t value)
{
    return (value != 0) && ((value & (value - 1)) == 0);
}

/*
Return the next power of 2 for provided value. If value is already power of 2, it returns value.
*/
uint_t nextPowerOf2(uint_t value)
{
    if (isPowerOfTwo(value))
    {
        return value;
    }

    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;

    return value;
}

/*
Returns the previous power of 2 for provided value. If value is already power of 2, it returns value.
*/
uint_t previousPowerOf2(uint_t value)
{
    if (isPowerOfTwo(value))
    {
        return value;
    }

    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value -= value >> 1;

    return value;
}

// Rounds the number to next multiple of provided value
int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
    {
        return numToRound;
    }

    int remainder = numToRound % multiple;

    if (remainder == 0)
    {
        return numToRound;
    }

    return numToRound + multiple - remainder;
}

/*
According to provided distribution returns distribution name.
*/
char* getDistributionName(data_dist_t distribution)
{
    switch (distribution)
    {
        case DISTRIBUTION_UNIFORM: return "uniform";
        case DISTRIBUTION_GAUSSIAN: return "gaussian";
        case DISTRIBUTION_ZERO: return "zero";
        case DISTRIBUTION_BUCKET: return "bucket";
        case DISTRIBUTION_STAGGERED: return "staggered";
        case DISTRIBUTION_SORTED_ASC: return "sorted_asc";
        case DISTRIBUTION_SORTED_DESC: return "sorder_desc";
        default:
            printf("Invalid distribution");
            exit(EXIT_FAILURE);
            return "";
    }
}

/*
Capitalizes string.
*/
std::string strCapitalize(std::string text)
{
    text[0] = toupper(text[0]);
    return text;
}

/*
Replaces specified characters in string.
*/
std::string strReplace(std::string text, char from, char to)
{
    for (int i = 0; i < text.length(); i++)
    {
        if (text[i] == from)
        {
            text[i] = to;
        }
    }

    return text;
}

/*
Replaces all spaces with '_'.
*/
std::string strSlugify(std::string text)
{
    return strReplace(text, ' ', '_');
}
