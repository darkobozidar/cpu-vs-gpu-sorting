#ifndef CONSTANTS_H
#define CONSTANTS_H

// Use multistep kernel with REGISTERS or SHARED MEMORY. Also sets different parameters (bellow).
#define USE_REGISTERS_MULTISTEP 0

#if USE_REGISTERS_MULTISTEP
    #define THREADS_PER_MERGE 512
    #define MAX_THREADS_PER_MULTISTEP 128
    #define MAX_MULTI_STEP 5
#else
    #define THREADS_PER_MERGE 512
    #define MAX_THREADS_PER_MULTISTEP 128
    #define MAX_MULTI_STEP 4
#endif

#endif
