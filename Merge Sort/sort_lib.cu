#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types.h"


//__host__ __device__ int_t compare(const void* elem1, const void* elem2) {
//    return (*(data_t*)elem1 - *(data_t*)elem2);
//}

//__device__ void compareExchange(data_t elem1, data_t elem2, bool orderAsc) {
//    /*printf("%d %d %d\n", *elem1, *elem1, orderAsc);*/
//
//    //bool comparison = 1;// (compare(elem1, elem2) > 0) ^ (!orderAsc);
//
//    /*data_t temp = comparison * (*elem1) + (!comparison) * (*elem2);
//    *elem1 = comparison * (*elem2) + (!comparison) * temp;
//    *elem2 = comparison * temp + (!comparison) * (*elem1);*/
//}
