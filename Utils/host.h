#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <windows.h>
#include <string>

void startStopwatch(LARGE_INTEGER* start);
double endStopwatch(LARGE_INTEGER start, char* comment);
double endStopwatch(LARGE_INTEGER start);
bool compareArrays(data_t* array1, data_t* array2, uint_t arrayLen);
void printTable(data_t *table, uint_t tableLen);
void printTable(data_t *table, uint_t startIndex, uint_t endIndex);
void checkMallocError(void *ptr);
bool isPowerOfTwo(uint_t value);
uint_t nextPowerOf2(uint_t value);
uint_t previousPowerOf2(uint_t value);
int roundUp(int numToRound, int multiple);
char* getDistributionName(data_dist_t distribution);
std::string strCapitalize(std::string str);
std::string strReplace(std::string text, char from, char to);
std::string strSlugify(std::string text);

#endif
