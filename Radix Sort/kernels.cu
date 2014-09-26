#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <climits>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_types.h"
#include "constants.h"


__global__ void sortBlockKernel(el_t *table, uint_t digit, bool orderAsc) {
    extern __shared__ el_t sortTile;

    for (uint_t offset = digit * BIT_COUNT; offset < digit * (BIT_COUNT + 1); offset++) {

    }
}
