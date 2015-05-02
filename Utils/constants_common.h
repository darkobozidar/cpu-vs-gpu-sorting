#ifndef CONSTANTS_COMMON_H
#define CONSTANTS_COMMON_H


/* ------------ GENERAL DEVICE PARAMETERS ----------- */
// These constants are needed in order to run C++ "templates", because variables cannot be used.

// How many threads are in warp (depending on device - for future compatibility)
#define WARP_SIZE 32
// Log¡2 of WARP_SIZE for faster computation because of left/right bit-shifts
#define WARP_SIZE_LOG 5

#endif
