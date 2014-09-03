#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <windows.h>

void startStopwatch(LARGE_INTEGER* start);
void endStopwatch(LARGE_INTEGER start, char *comment, char deviceType);
void endStopwatch(LARGE_INTEGER start, char *comment);
void fillTable(el_t *table, uint_t tableLen, uint_t interval);
void compareArrays(data_t* array1, data_t* array2, uint_t arrayLen);
void printTable(el_t *table, uint_t tableLen);
void printTable(el_t *table, uint_t startIndex, uint_t endIndex);

#endif
