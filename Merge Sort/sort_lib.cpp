#include "data_types.h"


int_t compare(const void * elem1, const void * elem2) {
    return (*(data_t*)elem1 - *(data_t*)elem2);
}
