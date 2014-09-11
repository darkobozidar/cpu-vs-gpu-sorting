#include <stdio.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"
#include "constants.h"
#include "utils_cuda.h"
#include "utils_host.h"


void sortParallel(el_t *h_input, el_t *h_output, uint_t tableLen, bool orderAsc) {
    printf("TODO parallel implementation\n");
}
