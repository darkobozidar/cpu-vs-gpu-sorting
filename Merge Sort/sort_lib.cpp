#include "data_types.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int_t compare(const void * elem1, const void * elem2) {
    return (*(data_t*)elem1 - *(data_t*)elem2);
}

__device__ void compareExchange(data_t* elem1, data_t elem2, bool orderAsc) {

}
