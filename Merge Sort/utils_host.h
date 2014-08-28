#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <windows.h>

void startStopwatch(LARGE_INTEGER* start);
void endStopwatch(LARGE_INTEGER start, char *comment, char deviceType);
void endStopwatch(LARGE_INTEGER start, char *comment);
void fillArrayRand(data_t* array, uint_t arrayLen);
void fillArrayValue(data_t* table, uint_t tableLen, data_t value);
void compareArrays(data_t* array1, data_t* array2, uint_t arrayLen);
void printArray(data_t* array, uint_t arrrayLen);
data_t* copyArray(data_t* array, uint_t arrayLen);

#endif
