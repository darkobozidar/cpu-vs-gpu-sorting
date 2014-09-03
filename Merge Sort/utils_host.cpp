#include <stdio.h>
#include <stdint.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"


void startStopwatch(LARGE_INTEGER* start) {
    QueryPerformanceCounter(start);
}

void endStopwatch(LARGE_INTEGER start, char* comment, char deviceType) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER end;
    double elapsedTime;
    char* device;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&end);
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    if (deviceType == 'H' || deviceType == 'h') {
        device = "HOST   >>> ";
    } else if (deviceType == 'D' || deviceType == 'd') {
        device = "DEVICE >>> ";
    } else if (deviceType == 'M' || deviceType == 'm') {
        device = "MEMCPY >>> ";
    } else {
        device = "";
    }

    printf("%s%s: %.5lf ms\n", device, comment, elapsedTime);
}

void endStopwatch(LARGE_INTEGER start, char* comment) {
    endStopwatch(start, comment, NULL);
}

void fillArrayRand(data_t* table, uint_t tableLen, uint_t interval) {
    for (uint_t i = 0; i < tableLen; i++) {
        table[i] = rand() % interval;
    }
}

void fillArrayConsecutive(data_t* table, uint_t tableLen) {
    for (uint_t i = 0; i < tableLen; i++) {
        table[i] = i;
    }
}

void fillArrayValue(data_t* table, uint_t tableLen, data_t value) {
    for (uint_t i = 0; i < tableLen; i++) {
        table[i] = value;
    }
}

void compareArrays(data_t* array1, data_t* array2, uint_t arrayLen) {
    for (uint_t i = 0; i < arrayLen; i++) {
        if (array1[i] != array2[i]) {
            printf("Arrays are different: array1[%d] = %d, array2[%d] = %d.\n", i, array1[i], i, array2[i]);
            return;
        }
    }

    printf("Arrays are the same.\n");
}

void printArray(data_t* array, uint_t arrayLen) {
    for (uint_t i = 0; i < arrayLen; i++) {
        char* separator = i == arrayLen - 1 ? "" : ", ";
        printf("%d%s", array[i], separator);
    }

    printf("\n\n");
}

void printArray(data_t* array, uint_t startIndex, uint_t endIndex) {
    for (uint_t i = startIndex; i <= endIndex; i++) {
        char* separator = i == endIndex ? "" : ", ";
        printf("%d%s", array[i], separator);
    }

    printf("\n\n");
}

data_t* copyArray(data_t* array, uint_t arrayLen) {
    data_t* arrayCopy = (data_t*)malloc(arrayLen * sizeof(*arrayCopy));

    for (int i = 0; i < arrayLen; i++) {
        arrayCopy[i] = array[i];
    }

    return arrayCopy;
}
