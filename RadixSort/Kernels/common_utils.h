#ifndef KERNELS_COMMON_UTILS_RADIX_SORT_H
#define KERNELS_COMMON_UTILS_RADIX_SORT_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"


/*
Generates lane mask needed to calculate warp scan of predicates.
*/
inline __device__ uint_t laneMask()
{
    uint_t mask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    return mask;
}

/*
Performs scan for each warp.
*/
inline __device__ uint_t binaryWarpScan(bool pred)
{
    uint_t mask = laneMask();
    uint_t ballot = __ballot(pred);
    return __popc(ballot & mask);
}

#endif
