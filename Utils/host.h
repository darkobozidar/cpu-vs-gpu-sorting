#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <windows.h>

void startStopwatch(LARGE_INTEGER* start);
double endStopwatch(LARGE_INTEGER start, char* comment);
double endStopwatch(LARGE_INTEGER start);
void fillTable(el_t *table, uint_t tableLen, uint_t interval);
void compareArrays(el_t* array1, el_t* array2, uint_t arrayLen);
void printTable(el_t *table, uint_t tableLen);
void printTable(el_t *table, uint_t startIndex, uint_t endIndex);
void checkMallocError(void *ptr);
uint_t nextPowerOf2(uint_t value);

#endif
