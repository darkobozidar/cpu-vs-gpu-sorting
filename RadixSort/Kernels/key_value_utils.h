#ifndef KERNELS_KEY_VALUE_UTILS_RADIX_SORT_H
#define KERNELS_KEY_VALUE_UTILS_RADIX_SORT_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/constants_common.h"
#include "../constants.h"
#include "common_utils.h"


/*
Performs scan for provided predicates and returns structure of results for each predicate.
The function is the same as in key-only version. The only difference is the preprocessor constant value
ELEMS_LOCAL_KV.
*/
template <uint_t blockSize>
__device__ uint_t intraBlockScanKeyValue(
#if (ELEMS_LOCAL_KV >= 1)
    bool pred0
#endif
#if (ELEMS_LOCAL_KV >= 2)
    , bool pred1
#endif
#if (ELEMS_LOCAL_KV >= 3)
    , bool pred2
#endif
#if (ELEMS_LOCAL_KV >= 4)
    , bool pred3
#endif
#if (ELEMS_LOCAL_KV >= 5)
    , bool pred4
#endif
#if (ELEMS_LOCAL_KV >= 6)
    , bool pred5
#endif
#if (ELEMS_LOCAL_KV >= 7)
    , bool pred6
#endif
#if (ELEMS_LOCAL_KV >= 8)
    , bool pred7
#endif
    )
{
    extern __shared__ uint_t scanTile[];
    uint_t warpIdx = threadIdx.x / WARP_SIZE;
    uint_t laneIdx = threadIdx.x & (WARP_SIZE - 1);
    uint_t warpResult = 0;
    uint_t predResult = 0;

#if (ELEMS_LOCAL_KV >= 1)
    warpResult += binaryWarpScan(pred0);
    predResult += pred0;
#endif
#if (ELEMS_LOCAL_KV >= 2)
    warpResult += binaryWarpScan(pred1);
    predResult += pred1;
#endif
#if (ELEMS_LOCAL_KV >= 3)
    warpResult += binaryWarpScan(pred2);
    predResult += pred2;
#endif
#if (ELEMS_LOCAL_KV >= 4)
    warpResult += binaryWarpScan(pred3);
    predResult += pred3;
#endif
#if (ELEMS_LOCAL_KV >= 5)
    warpResult += binaryWarpScan(pred4);
    predResult += pred4;
#endif
#if (ELEMS_LOCAL_KV >= 6)
    warpResult += binaryWarpScan(pred5);
    predResult += pred5;
#endif
#if (ELEMS_LOCAL_KV >= 7)
    warpResult += binaryWarpScan(pred6);
    predResult += pred6;
#endif
#if (ELEMS_LOCAL_KV >= 8)
    warpResult += binaryWarpScan(pred7);
    predResult += pred7;
#endif
    __syncthreads();

    if (laneIdx == WARP_SIZE - 1)
    {
        scanTile[warpIdx] = warpResult + predResult;
    }
    __syncthreads();

    // Maximum number of elements for scan is warpSize ^ 2
    if (threadIdx.x < WARP_SIZE)
    {
        scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
    }
    __syncthreads();

    return warpResult + scanTile[warpIdx];
}

#endif
