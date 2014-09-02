//#include <stdio.h>
//
//#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include "data_types.h"
//
//
///*
//For debugging purposes only specified thread prints to console.
//*/
//__device__ void printOnce(char* text, uint_t threadIndex) {
//    if (threadIdx.x == threadIndex) {
//        printf(text);
//    }
//}
//
///*
//For debugging purposes only thread 0 prints to console.
//*/
//__device__ void printOnce(char* text) {
//    printOnce(text, 0);
//}
